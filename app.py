import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import box
from shapely.ops import unary_union
import pandas as pd
from deepforest import main, get_data

st.set_page_config(page_title="CanopyScope - Carbon dMRV", layout="wide")

@st.cache_resource
def load_detector():
    model = main.deepforest()
    try:
        model.use_release()
    except AttributeError:
        model.load_model("weecology/deepforest-tree")
    return model

st.title("🌲 CanopyScope: Individual Tree Crown & Canopy dMRV")
st.markdown("""
Verifiable tree crown detection and overlap-deduplicated canopy quantification for digital MRV in voluntary carbon markets.
""")

# Sidebar settings
st.sidebar.header("Controls & Calibration")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 0.80, 0.25, 0.05)
resolution_cm = st.sidebar.number_input("Pixel Resolution (GSD in cm/px)", 5.0, 50.0, 10.0)

source_choice = st.sidebar.radio("Image Source", ["Built-in NEON Benchmark (OSBS)", "Upload Aerial/Satellite Tile"])

img = None
if source_choice == "Built-in NEON Benchmark (OSBS)":
    sample_path = get_data("OSBS_029.png")
    img = Image.open(sample_path).convert("RGB")
else:
    uploaded = st.sidebar.file_uploader("Upload Image (PNG, JPG, TIF)", type=["png", "jpg", "jpeg", "tif"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")

if img is not None:
    img_array = np.array(img)
    h, w, _ = img_array.shape

    with st.spinner("Executing RetinaNet inference & spatial polygon union..."):
        detector = load_detector()
        # Save temp image for native v2 path prediction
        temp_path = "temp_input.png"
        img.save(temp_path)
        preds = detector.predict_image(path=temp_path)

    df_trees = preds[preds["score"] >= conf_thresh].copy() if (preds is not None and not preds.empty) else pd.DataFrame()
    tree_count = len(df_trees)

    # Spatial Geometry: Ground meter conversion
    m_per_px = resolution_cm / 100.0
    polygons = [
        box(r["xmin"] * m_per_px, r["ymin"] * m_per_px, r["xmax"] * m_per_px, r["ymax"] * m_per_px)
        for _, r in df_trees.iterrows()
    ]
    
    total_aoi_m2 = (w * m_per_px) * (h * m_per_px)
    if polygons:
        canopy_union = unary_union(polygons)
        # Apply pi/4 geometric factor to correct rectangular bounding-box overestimation
        total_canopy_m2 = canopy_union.area * 0.785
    else:
        total_canopy_m2 = 0.0

    canopy_cover_pct = (total_canopy_m2 / total_aoi_m2 * 100) if total_aoi_m2 > 0 else 0

    # Display KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected Stems", f"{tree_count:,}")
    c2.metric("Effective Canopy Area", f"{total_canopy_m2:.1f} m²")
    c3.metric("Hectares", f"{(total_canopy_m2 / 10000):.4f} ha")
    c4.metric("Canopy Cover", f"{canopy_cover_pct:.1f}%")

    # Visual Plot
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img_array)
    for _, r in df_trees.iterrows():
        rect = patches.Rectangle(
            (r["xmin"], r["ymin"]), r["xmax"] - r["xmin"], r["ymax"] - r["ymin"],
            linewidth=1.3, edgecolor="cyan", facecolor="none"
        )
        ax.add_patch(rect)
    ax.axis("off")
    st.pyplot(fig)

    # Downloadable CSV Audit Log for Carbon Auditors
    if not df_trees.empty:
        csv_data = df_trees[["xmin", "ymin", "xmax", "ymax", "score"]].to_csv(index=False)
        st.download_button("📥 Export Crown Audit Log (CSV)", csv_data, "canopy_audit_log.csv", "text/csv")
else:
    st.info("Select a preset or upload an image from the sidebar.")