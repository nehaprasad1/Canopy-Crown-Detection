import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from deepforest import main, get_data

st.set_page_config(page_title="CanopyScope dMRV", layout="wide")

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
# Hierarchical Containment & De-Clustering Engine
# ---------------------------------------------------------
def resolve_dense_canopy_clusters(df, containment_threshold=0.55):
    """
    Suppresses micro-boxes that are fragmented pieces of larger primary crowns.
    If a smaller candidate box overlaps heavily inside a larger candidate box,
    it is dropped to eliminate branch over-segmentation.
    """
    if df.empty or len(df) <= 1:
        return df

    # Sort boxes by area descending (largest crowns first)
    df["area_px"] = (df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])
    df = df.sort_values(by="area_px", ascending=False).reset_index(drop=True)

    keep_indices = []
    polys = [box(r["xmin"], r["ymin"], r["xmax"], r["ymax"]) for _, r in df.iterrows()]

    for i in range(len(df)):
        keep = True
        for j in keep_indices:
            # Check intersection area over the smaller box's area
            inter_area = polys[i].intersection(polys[j]).area
            if inter_area / polys[i].area > containment_threshold:
                keep = False
                break
        if keep:
            keep_indices.append(i)

    return df.iloc[keep_indices].copy().reset_index(drop=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🌲 CanopyScope dMRV")
st.sidebar.subheader("1. Source Imagery")

input_mode = st.sidebar.radio("Input Source", ["Preloaded Benchmarks", "Upload Custom Orthomosaic"])
raw_img = None

if input_mode == "Preloaded Benchmarks":
    site = st.sidebar.selectbox("Select Preset", ["SOAP Montane Forest", "OSBS Conifer Pine"])
    filename = "SOAP_061.png" if "SOAP" in site else "OSBS_029.png"
    raw_img = Image.open(get_data(filename)).convert("RGB")
else:
    uploaded = st.sidebar.file_uploader("Upload Drone Orthomosaic (.png, .jpg, .tif)", type=["png", "jpg", "jpeg", "tif"])
    if uploaded is not None:
        raw_img = Image.open(uploaded).convert("RGB")
    else:
        raw_img = Image.open(get_data("SOAP_061.png")).convert("RGB")
        st.sidebar.info("Using default stand until a file is uploaded.")

st.sidebar.subheader("2. Quality & Gating Controls")
conf_thresh = st.sidebar.slider("Detector Confidence", 0.15, 0.60, 0.30, 0.05)
enable_declustering = st.sidebar.toggle("De-Cluster Dense Foliage (Suppress Branch Fragments)", value=True)
enable_ndvi = st.sidebar.toggle("Disqualify Deadwood (NDVI)", value=True)
ndvi_thresh = st.sidebar.slider("NDVI Cutoff", 0.05, 0.35, 0.18, 0.01)
gsd_cm = st.sidebar.number_input("Resolution (GSD cm/px)", 5.0, 50.0, 10.0, step=1.0)
m_per_px = gsd_cm / 100.0

# ---------------------------------------------------------
# Core Analytics Pipeline
# ---------------------------------------------------------
img_arr = np.array(raw_img)
h, w, _ = img_arr.shape
total_aoi_m2 = (w * m_per_px) * (h * m_per_px)

# Radiometric NIR Simulation & NDVI Surface
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

# Neural Network Detection
temp_path = "temp_tile.png"
raw_img.save(temp_path)
with st.spinner("Analyzing canopy structures..."):
    preds = detector.predict_image(path=temp_path)

df_trees = preds[preds["score"] >= conf_thresh].copy() if (preds is not None and not preds.empty) else pd.DataFrame()

# Apply Dense Canopy De-clustering
if not df_trees.empty and enable_declustering:
    df_trees = resolve_dense_canopy_clusters(df_trees, containment_threshold=0.50)

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
    df_trees["est_diameter_m"] = ((df_trees["xmax"] - df_trees["xmin"]) + (df_trees["ymax"] - df_trees["ymin"])) / 2.0 * m_per_px

    if enable_ndvi:
        df_live = df_trees[df_trees["mean_ndvi"] >= ndvi_thresh].copy()
        df_dead = df_trees[df_trees["mean_ndvi"] < ndvi_thresh].copy()
    else:
        df_live = df_trees.copy()
        df_dead = pd.DataFrame()

    # Spatial Deduplication
    polygons_live_m = [box_to_polygon(r, scale=m_per_px) for _, r in df_live.iterrows()]
    polygons_live_px = [box_to_polygon(r, scale=1.0) for _, r in df_live.iterrows()]
    polygons_dead_px = [box_to_polygon(r, scale=1.0) for _, r in df_dead.iterrows()] if not df_dead.empty else []

    live_canopy_m2 = unary_union(polygons_live_m).area if polygons_live_m else 0.0
    canopy_cover_pct = (live_canopy_m2 / total_aoi_m2) * 100.0

    # Main Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌲 Verified Stems", f"{len(df_live)} stems")
    c2.metric("🌿 Audited Canopy Area", f"{live_canopy_m2:.1f} m²", f"{(live_canopy_m2/10000):.4f} ha")
    c3.metric("📊 Canopy Coverage", f"{canopy_cover_pct:.1f}%")
    c4.metric("🛡️ Snags Disqualified", f"{len(df_dead)} snags", delta_color="inverse")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Crown Delineation")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.imshow(img_arr)
        for p in polygons_live_px:
            ax1.add_patch(MplPolygon(np.array(p.exterior.coords), closed=True, edgecolor="#00f2fe", facecolor="none", linewidth=1.8))
        for p in polygons_dead_px:
            ax1.add_patch(MplPolygon(np.array(p.exterior.coords), closed=True, edgecolor="#ff4b4b", linestyle="--", facecolor="none", linewidth=1.8))
        ax1.axis("off")
        st.pyplot(fig1, use_container_width=True)
        st.caption("🔷 **Cyan**: Living Canopy &nbsp;|&nbsp; 🔴 **Red**: Disqualified Snags/Ground")

    with col2:
        st.subheader("2. Photosynthetic Surface")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        im = ax2.imshow(ndvi_surface, cmap="RdYlGn", vmin=-0.1, vmax=0.6)
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis("off")
        st.pyplot(fig2, use_container_width=True)
        st.caption("Chlorophyll verification layer auditing inner crown cores.")
