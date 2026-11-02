# Phase 0: Dataset Inventory & Integrity Audit

**Date:** 2026-11-01  
**Project:** Traffic Intelligence AI  
**Archive Location:** `E:\Neural_Network_project\Data`

---

## 1. Archives Overview

| Archive Name | File Size | Internal Format | Verification Status | Primary Intended Task |
| :--- | :---: | :--- | :---: | :--- |
| **`archive (1).zip`** | 23.24 GB | TuSimple (train_set + test_set + seg_label) | **Verified 100% Intact** | Lane Detection & Segmentation (Module A) |
| **`bdd100k_drivable_maps.zip`** | 0.47 GB | PNG binary mask images (train + val) | **Verified 100% Intact** | Drivable Area Segmentation (Module C) |
| **`bdd100k_images_100k.zip`** | 5.67 GB | JPEG 1280×720 road frames (train + val + test) | **Verified 100% Intact** | Road & Vehicle Context (Module B & C) |
| **`bdd100k_det_20_labels.zip`** | 0.23 GB | 2,000 JPEG video sample frames (`-0000100.jpg`) | **Missing JSON Bounding Boxes** | Bounding Box Detection (Module B) - Requires JSON |

---

## 2. Detailed Dataset Breakdown

### A. TuSimple Lane Detection Dataset (`archive (1).zip` — 23.24 GB)
- **Archive Size:** 23,235,794,837 bytes (~23.24 GB)
- **Total Entries inside Zip:** 134,580 files
- **Clip Image Frames:**
  - Training clip images: **72,520 frames** (in 20-frame continuous clips under `TUSimple/train_set/clips/`)
  - Testing clip images: **55,640 frames** (under `TUSimple/test_set/clips/`)
- **Pixel-Level Segmentation Masks (`seg_label`):**
  - **6,408 mask PNGs** matching the annotated 20th keyframe of every clip.
- **Coordinate Annotations (JSON Lines):**
  - `TUSimple/train_set/label_data_0313.json`: **2,858 annotated frames**
  - `TUSimple/train_set/label_data_0531.json`: **358 annotated frames**
  - `TUSimple/train_set/label_data_0601.json`: **410 annotated frames**
  - `TUSimple/test_label.json`: **2,782 annotated frames**
  - **Total Training Annotated Frames:** 2,858 + 358 + 410 = **3,626 frames**
  - **Total Annotated Frames (Train + Test):** 3,626 + 2,782 = **6,408 frames** (Matches `seg_label` count exactly!)
- **Annotation Format:**
  ```json
  {
    "raw_file": "clips/0313-1/6040/20.jpg",
    "lanes": [[-2, -2, 580, 582, ...], ...],
    "h_samples": [240, 250, 260, ...]
  }
  ```
- **Conclusion:** **TuSimple is 100% complete and self-contained.** Both coordinate prediction and semantic segmentation models can be trained immediately.

---

### B. BDD100K Drivable Maps (`bdd100k_drivable_maps.zip` — 0.47 GB)
- **Total Entries inside Zip:** 160,000 PNG files
- **Labels (Index Masks):**
  - `labels/train/`: **70,000 PNGs**
  - `labels/val/`: **10,000 PNGs**
- **Color Labels (Visual Verification Masks):**
  - `color_labels/train/`: **70,000 PNGs**
  - `color_labels/val/`: **10,000 PNGs**
- **Mask Properties:**
  - Dimensions: 1280 × 720, single channel `uint8`
  - Values: `0` (background/non-drivable), `1` (drivable area)
- **Conclusion:** **100% complete and ready.**

---

### C. BDD100K 100k Images (`bdd100k_images_100k.zip` — 5.67 GB)
- **Total Entries inside Zip:** 100,000 JPEG files
- **Dimensions:** 1280 × 720
- **Cross-Check with Drivable Maps:**
  - Exactly **80,000 images** match the drivable maps (70,000 train + 10,000 val).
- **Conclusion:** **100% complete and pairs directly with drivable maps.**

---

### D. BDD100K Detection Labels (`bdd100k_det_20_labels.zip` — 0.23 GB)
- **Total Entries inside Zip:** 2,000 JPEG images
  - `train/`: 1,400 images
  - `val/`: 200 images
  - `test/`: 400 images
- **Missing Elements:**
  - This archive contains `.jpg` frame samples, **NOT** the JSON bounding box annotations (`category`, `box2d`: `[x1, y1, x2, y2]`).
- **Required Action:**
  - We need the BDD100K detection labels JSON file (e.g. `bdd100k_det_20_labels_trainval.json` or detection annotation file, typically ~15–25 MB uncompressed JSON).
