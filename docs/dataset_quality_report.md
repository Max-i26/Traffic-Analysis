# Phase 0: Scope & Dataset Adequacy Analysis

**Project Title:** Neural Network-Based Traffic Scene Understanding, Lane Detection and Temporal Behaviour Analysis  
**Repository:** [https://github.com/Max-i26/Traffic-Analysis](https://github.com/Max-i26/Traffic-Analysis)

---

## 1. Project Scopes vs. Dataset Adequacy Matrix

| # | Project Scope / Module | Required Data & Modality | Available in Current Datasets? | Adequacy & Action Plan |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Module A: Lane Detection & Segmentation** | Road images + lane coordinates + segmentation masks | **YES (TuSimple)** | **100% Sufficient.** `archive (1).zip` contains 6,408 annotated frames with coordinate JSONs and segmentation mask images (`seg_label`). Ready for CNN and U-Net training. |
| **2** | **Module B: Vehicle Detection** | Road images + 2D bounding boxes (car, bus, truck, motorcycle) | **PARTIAL** | **Images present (100k images), but bounding-box JSON is missing.** `bdd100k_det_20_labels.zip` has 2k sample JPEG frames instead of JSON annotations. Need BDD100K detection labels JSON file (~20 MB). |
| **3** | **Module C: Road / Drivable Area Segmentation** | Road images + pixel segmentation masks | **YES (BDD100K)** | **100% Sufficient.** 80,000 paired images and drivable masks (70k train, 10k val) are verified and intact. |
| **4** | **Module D: Vehicle Tracking & Trajectory** | Sequential video frames + detection bounding boxes | **YES (TuSimple clips & BDD video frames)** | **100% Sufficient.** TuSimple contains 128,000+ continuous clip frames (20 consecutive frames per clip), perfect for multi-object tracking (ByteTrack / SORT) and trajectory generation. |
| **5** | **Module E: Temporal Traffic Behaviour & Violations** | Multi-frame trajectory vectors + lane associations + event labels | **RULE / HYBRID DERIVATION** | **Master Prompt Strategy:** Standard datasets (TuSimple / BDD100K) do not label legal violations. As per research spec, we construct temporal trajectory sequences (displacement, direction, lane crossing) and validate with rule-based heuristics and synthetic/annotated event sequences. |
| **6** | **Module F: Streamlit Application Dashboard** | Model weights + video test clips + UI components | **YES** | **100% Sufficient.** Video clips from TuSimple and test sets serve as real evaluation footage. |

---

## 2. Summary of Findings

1. **Are the datasets enough to make the project?**
   - **YES, for Modules A, C, D, E, and F.**
   - For **Module B (Vehicle Detection)**, we have 100,000 raw road images, but we need the small JSON file containing bounding boxes (`bdd100k_det_20_labels_trainval.json` or BDD100K 100k detection JSON).

2. **Can we start immediately without waiting?**
   - **YES.** In the Master Project Prompt, **Module A (Lane Detection & Segmentation)** is specified as our **first priority**.
   - TuSimple has all required images, JSON lane coordinates, and segmentation ground-truth masks. We can complete Phase 1, Phase 2, and Phase 3 entirely right away.

3. **Storage Strategy:**
   - Instead of extracting all 29 GB at once (which would expand to ~80 GB of uncompressed data), we will extract only the curated subsets needed for each phase into `datasets/extracted/` and `datasets/processed/`.
