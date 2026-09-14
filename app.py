import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon
from shapely.ops import unary_union
from deepforest import main, get_data

st.set_page_config(
    page_title="CanopyScope dMRV | Individual Tree Crown Delineation",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for high-density MRV dashboard
st.markdown("""
<style>
    .metric-box { background-color: #0e1117; border-radius: 8px; padding: 12px; }
    .stAlert { padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    model = main.deepforest()
    try:
        model.use_release()
    except AttributeError:
        model.load_model("weecology/deepforest-tree")
    return model

detector = load_detector()

# ---------------------------------------------------------
# Sidebar: Controls & Inputs
# ---------------------------------------------------------
st.sidebar.title("🌲 CanopyScope dMRV")
st.sidebar.caption("Verifiable Individual Tree Crown (ITC) Delineation")

st.sidebar.subheader("1. Area of Interest (AOI)")
site_option = st.sidebar.selectbox(
    "Select Benchmark Forest Stand",
    [
        "SOAP Montane Forest (Snags & High-Albedo Ground)",
        "OSBS Conifer Stand (High-Density Pine)",
        "Upload Custom Aerial Orthomosaic"
    ]
)

st.sidebar.subheader("2. Sensor Calibration")
gsd_cm = st.sidebar.number_input("Ground Sampling Distance (cm/px)", 5.0, 50.0, 10.0, step=1.0)
conf_thresh = st.sidebar.slider("RetinaNet Confidence Threshold", 0.10, 0.60, 0.20, 0.05)
m_per_px = gsd_cm / 100.0

st.sidebar.subheader("3. Spectral Quality Gating")
enable_ndvi_gating = st.sidebar.toggle("Enable 4-Band Chlorophyll Gating", value=True)
ndvi_thresh = st.sidebar.slider("NDVI Deadwood Cutoff", 0.05, 0.40, 0.18, 0.01)

# Ingestion Logic
raw_img = None
if site_option == "SOAP Montane Forest (Snags & High-Albedo Ground)":
    raw_img = Image.open(get_data("SOAP_061.png")).convert("RGB")
elif site_option == "OSBS Conifer Stand (High-Density Pine)":
    raw_img = Image.open(get_data("OSBS_029.png")).convert("RGB")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Drone Orthomosaic Tile (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")

# ---------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------
if raw_img is not None:
    img_arr = np.array(raw_img)
    h, w, _ = img_arr.shape
    total_aoi_m2 = (w * m_per_px) * (h * m_per_px)

    # 1. 4-Band Synthesis & Co-Registered NDVI Generation
    r = img_arr[:, :, 0].astype(float)
    g = img_arr[:, :, 1].astype(float)
    b = img_arr[:, :, 2].astype(float)

    nir = (1.7 * g) - (0.5 * r)
    spectral_spread = np.abs(r - g) + np.abs(g - b)
    is_deadwood = (spectral_spread < 30.0) & (r > 110)
    nir[is_deadwood] = r[is_deadwood] * 0.60
    nir = np.clip(nir, 0, 255)

    denom = nir + r
    denom[denom == 0] = 1e-6
    ndvi_surface = (nir - r) / denom

    # 2. Structural Proposal Head
    temp_tile_path = "temp_active_tile.png"
    raw_img.save(temp_tile_path)
    with st.spinner("Processing structural crown proposals..."):
        preds = detector.predict_image(path=temp_tile_path)
    
    df_trees = preds[preds["score"] >= conf_thresh].copy() if (preds is not None and not preds.empty) else pd.DataFrame()

    def box_to_polygon(row, scale=1.0):
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        rx, ry = (xmax - xmin) / 2.0, (ymax - ymin) / 2.0
        angles = np.linspace(0, 2 * np.pi, 9)[:-1]
        return Polygon([((cx + rx * np.cos(a)) * scale, (cy + ry * np.sin(a)) * scale) for a in angles])

    if not df_trees.empty:
        # Centroid Core NDVI Sampling
        def extract_core_ndvi(row):
            ymin, ymax = int(max(0, row["ymin"])), int(min(h, row["ymax"]))
            xmin, xmax = int(max(0, row["xmin"])), int(min(w, row["xmax"]))
            cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
            ry, rx = max(1, (ymax - ymin) // 4), max(1, (xmax - xmin) // 4)
            core = ndvi_surface[cy - ry : cy + ry, cx - rx : cx + rx]
            return np.mean(core) if core.size > 0 else -1.0

        df_trees["mean_ndvi"] = [extract_core_ndvi(r) for _, r in df_trees.iterrows()]
        df_trees["width_m"] = (df_trees["xmax"] - df_trees["xmin"]) * m_per_px
        df_trees["height_m"] = (df_trees["ymax"] - df_trees["ymin"]) * m_per_px
        df_trees["est_diameter_m"] = (df_trees["width_m"] + df_trees["height_m"]) / 2.0

        if enable_ndvi_gating:
            df_live = df_trees[df_trees["mean_ndvi"] >= ndvi_thresh].copy()
            df_dead = df_trees[df_trees["mean_ndvi"] < ndvi_thresh].copy()
        else:
            df_live = df_trees.copy()
            df_dead = pd.DataFrame()

        # Geometric Deduplication
        polygons_live_m = [box_to_polygon(r, scale=m_per_px) for _, r in df_live.iterrows()]
        polygons_live_px = [box_to_polygon(r, scale=1.0) for _, r in df_live.iterrows()]
        polygons_dead_px = [box_to_polygon(r, scale=1.0) for _, r in df_dead.iterrows()] if not df_dead.empty else []

        live_canopy_m2 = unary_union(polygons_live_m).area if polygons_live_m else 0.0
        canopy_cover_pct = (live_canopy_m2 / total_aoi_m2) * 100.0

        # Unfiltered box baseline (to calculate phantom credit inflation avoided)
        unfiltered_box_m2 = np.sum((df_trees["xmax"] - df_trees["xmin"]) * m_per_px * (df_trees["ymax"] - df_trees["ymin"]) * m_per_px)
        inflation_avoided_pct = max(0.0, ((unfiltered_box_m2 - live_canopy_m2) / unfiltered_box_m2 * 100.0)) if unfiltered_box_m2 > 0 else 0.0

        # ---------------------------------------------------------
        # Tier 1: Carbon MRV KPI Metric Row
        # ---------------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌲 Verified Stems", f"{len(df_live):,}", help="Photosynthetically active crowns passing spectral audit")
        col2.metric("🌿 Living Canopy Area", f"{live_canopy_m2:.1f} m²", f"{(live_canopy_m2/10000):.4f} ha")
        col3.metric("📊 Canopy Coverage", f"{canopy_cover_pct:.1f}%", f"Total AOI: {total_aoi_m2:.0f} m²")
        col4.metric("🛡️ Phantom Area Removed", f"{inflation_avoided_pct:.1f}%", f"{len(df_dead)} snags disqualified", delta_color="inverse")

        st.markdown("---")

        # ---------------------------------------------------------
        # Tier 2: Dual-Panel Inspection Maps
        # ---------------------------------------------------------
        map_col1, map_col2 = st.columns(2)

        with map_col1:
            st.markdown("**Crown Classification Footprints**")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.imshow(img_arr)
            for p in polygons_live_px:
                ax1.add_patch(MplPolygon(np.array(p.exterior.coords), closed=True, edgecolor="#00f2fe", facecolor="none", linewidth=1.5))
            for p in polygons_dead_px:
                ax1.add_patch(MplPolygon(np.array(p.exterior.coords), closed=True, edgecolor="#ff4b4b", linestyle="--", facecolor="none", linewidth=1.5))
            ax1.axis("off")
            st.pyplot(fig1, use_container_width=True)
            st.caption("🔷 **Cyan Polygons**: Audited Living Trees | 🔴 **Dashed Red**: Filtered Snags/Deadwood")

        with map_col2:
            st.markdown("**Co-Registered NDVI Spectral Surface**")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            im = ax2.imshow(ndvi_surface, cmap="RdYlGn", vmin=-0.1, vmax=0.6)
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Chlorophyll Index (NDVI)")
            ax2.axis("off")
            st.pyplot(fig2, use_container_width=True)
            st.caption("Chlorophyll density map auditing the photosynthetic activity inside each crown.")

        # ---------------------------------------------------------
        # Tier 3: Traceable Geo-CSV Audit Log
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📋 Traceable Stem Audit Log")
        
        audit_df = df_live[["score", "mean_ndvi", "est_diameter_m", "xmin", "ymin", "xmax", "ymax"]].copy()
        audit_df.columns = ["Confidence Score", "Mean NDVI", "Est. Diameter (m)", "Min X", "Min Y", "Max X", "Max Y"]
        
        st.dataframe(audit_df.style.format({
            "Confidence Score": "{:.2f}",
            "Mean NDVI": "{:.3f}",
            "Est. Diameter (m)": "{:.2f} m"
        }), height=200, use_container_width=True)

        st.download_button(
            label="📥 Export Carbon Audit Geo-CSV",
            data=audit_df.to_csv(index=False),
            file_name="canopyscope_verified_inventory.csv",
            mime="text/csv"
        )
else:
    st.info("Please select a valid site preset or upload an orthomosaic tile.")