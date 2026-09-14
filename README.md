# CanopyScope 🌲🛰️
### Verifiable Individual Tree Crown (ITC) Delineation & Biophysical Carbon dMRV

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://canopy-crown-detection-ehpqhyrhzuebqwqvqzdff.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CanopyScope** is an individual tree crown (ITC) digital Measurement, Reporting, and Verification (dMRV) pipeline designed for voluntary carbon markets (VCM). 

Standard computer vision detectors draw loose rectangular bounding boxes around trees, inflating canopy surface area by 10–15% and double-counting overlapping foliage. Furthermore, 3-band RGB models mistake dead snags, dry branches, and rocks for living biomass. CanopyScope bridges satellite remote sensing and conservative carbon accounting by combining **8-point polygonal regularization**, **Shapely spatial union merging**, and **centroid-core Near-Infrared (NDVI) chlorophyll gating**.

---

## 🔗 Quick Links
* **Live Streamlit Web Application:** [canopy-crown-detection-ehpqhyrhzuebqwqvqzdff.streamlit.app](https://canopy-crown-detection-ehpqhyrhzuebqwqvqzdff.streamlit.app/)
* **Author:** Neyhaa Prasad ([GitHub Profile](https://github.com/nehaprasad1))
* **Source Repository:** [github.com/nehaprasad1/Canopy-Crown-Detection](https://github.com/nehaprasad1/Canopy-Crown-Detection)

---

## 📌 Core Engineering Pipeline


[VHR Multi-Band Imagery (10cm GSD Benchmark Proxy)]
│
├───► [RGB Stream] ──► RetinaNet (ResNet-50) ──► 8-Point Polygonal Regularization
│                                                              │
└───► [NIR Stream] ──► Core NDVI Audit (NDVI ≥ 0.18) ──────────┘
│
[Photosynthetically Verified Crowns]
│
[Shapely unary_union Overlap Merging]
│
[Conservative tCO2e Carbon Ledger (Geo-CSV)]

1. **Structural Proposal Head:** Utilizes a pre-trained RetinaNet with ResNet-50 (`DeepForest`) calibrated to high-resolution canopy orthomosaics.
2. **8-Point Polygonal Regularizer:** Trims the 4 non-canopy corners of rectangular bounding boxes into organic octagons:
   $$P_k = \left( c_x + r_x \cos\left(\frac{2\pi k}{8}\right), \, c_y + r_y \sin\left(\frac{2\pi k}{8}\right) \right), \quad k \in \{0, \dots, 7\}$$
3. **Decoupled Core-NDVI Gating:** Samples the inner 50% radius core around the crown centroid to bypass ground grass bleed. Targets with `Core NDVI < 0.18` are flagged as non-photosynthetic necromass (deadwood/snags) and discarded.
4. **Spatial Overlap Deduplication:** Executes `shapely.ops.unary_union` across all verified live polygons, merging intersecting crowns into a single continuous boundary to completely eliminate double-counted canopy cover.
5. **Conservative Allometric Accounting:** Derives Aboveground Biomass (AGB) and CO₂ equivalent yields, applying a mandatory 15% risk buffer discount:
   $$\text{AGB (kg)} = 0.06 \cdot (\text{Area})^{1.35} \times 1000, \quad \text{tCO}_2\text{e} = \text{AGB} \times 0.47 \times 10^{-3} \times 3.67$$

---

## 📊 Benchmark Validation Findings

| Benchmark Site | Raw Optical Baseline | CanopyScope Audited | Audit & Carbon Impact |
| :--- | :--- | :--- | :--- |
| **NEON OSBS (Pine Stand)** | 536.2 m² (33.5% cover) | **488.2 m² (30.5% cover)** | Eliminated **9.8% phantom area inflation** from rectangular boxes. |
| **NEON SOAP (Montane Snags)** | 36 proposed stems (320.1 m²) | **7 verified stems (66.6 m²)** | Disqualified 29 dead snags/rocks; eliminated **79.2% phantom credit risk**. |

---

## ⚠️ Known Limitations & Failure Modes

Honesty about system boundaries is critical in carbon accounting:

* **Closed-Canopy Broadleaf Saturation (2D Planar Failure):** In dense deciduous forests where interlocking crowns form an uninterrupted canopy without optical shadow margins, 2D detectors suffer from crown clumping (under-segmentation) and branch fragmentation (over-segmentation). Resolving this requires 3D elevation gradients (LiDAR / stereo DSM).
* **Resolution & Anchor Coupling (Low-Res Breakdown):** The detector's feature pyramid anchors are calibrated for high-resolution imagery (~10 cm/px). When tested on downsampled, compressed imagery (>50 cm/px, such as a 21.6 KB web preview), individual crowns collapse below the receptive field, triggering spurious micro-boxes on leaf texture clusters.
* **Single-Epoch Phenology Blindness:** Single-date optical NDVI relies on active chlorophyll. During deciduous winter leaf-off conditions or seasonal dry spells, dormant living trees drop foliage and are incorrectly flagged as dead snags (`NDVI < 0.18`), necessitating multi-temporal time-series verification.
* **Absence of Vertical Height Allometry:** Biomass is calculated purely from 2D crown surface area without vertical tree height ($H$) or species wood density ($\rho$), introducing variance across stands of differing vertical maturity.
* **Synthetic Near-Infrared in Web Demo:** To maintain a lightweight browser runtime without streaming multi-gigabyte 16-bit GeoTIFF bands, the Near-Infrared surface is dynamically approximated using calibrated visible-band transformations and achromatic deadwood suppression.

---

## 🛠️ Local Setup & Installation

```bash
# 1. Clone repository
git clone [https://github.com/nehaprasad1/Canopy-Crown-Detection.git](https://github.com/nehaprasad1/Canopy-Crown-Detection.git)
cd Canopy-Crown-Detection

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Streamlit app
streamlit run app.py

🗺️ Production Engineering Roadmap
 * [ ] Phase 1 — Stereo-Satellite DSM & Spaceborne GEDI LiDAR: Fuse stereo-satellite surface models and NASA GEDI waveform transects to infer canopy height gradients (H) without airborne flights.
 * [ ] Phase 2 — Bivariate Height-Area Allometry: Upgrade biomass modeling to \text{AGB} = 0.0673 \cdot (\rho \cdot \text{Area} \cdot H)^{0.976}.
 * [ ] Phase 3 — Native 16-Bit Multispectral Ingestion: Direct ingestion of native calibrated surface reflectance rasters (PlanetScope 8-band, WorldView-3) with red-edge bands.
 * [ ] Phase 4 — Multi-Temporal Additionality Ledger: Automate annual differential surface auditing (T_{\text{year}} - T_{\text{baseline}}) with cryptographic audit hashes for registry issuance.
📝 Note on Project Origin & Hackathon Context
> Built for the Flora Carbon Tech Internship Challenge (September 2026)
> This project was developed as a solo submission for the weekend hackathon hosted by Flora Carbon for their Kolkata engineering internship track.
> The challenge evaluated working prototypes that can accurately delineate individual tree crowns, compute canopy area, and present actionable results while being completely transparent about edge cases, shortcuts, and physical limitations. Rather than inflating performance with vanity metrics, CanopyScope was built around the core ethos of carbon markets: conservatism, physical defensibility, and honest accounting over blind automation.
>  * Live Platform: https://canopy-crown-detection-ehpqhyrhzuebqwqvqzdff.streamlit.app/
>  * Solo Builder: Neyhaa Prasad
> 
📄 License
Distributed under the MIT License. See LICENSE for more information.

