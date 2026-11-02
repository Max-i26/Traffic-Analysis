# Phase 0: Environment & Hardware Audit Report

**Date:** 2026-11-01  
**Project:** Traffic Intelligence AI  
**Repository:** [https://github.com/Max-i26/Traffic-Analysis](https://github.com/Max-i26/Traffic-Analysis)

---

## 1. Hardware Audit

| Component | Detected Specification | Evaluation & Suitability |
| :--- | :--- | :--- |
| **GPU Model** | **NVIDIA GeForce RTX 2050 Laptop GPU** | Supported (Ampere architecture, Compute Capability 8.6) |
| **VRAM** | **4096 MiB (4.0 GB) GDDR6** | Constrained. Requires small batch sizes (2–4), mixed-precision (FP16/AMP), gradient accumulation, and moderate input resolutions (e.g. 512×288). |
| **GPU Driver** | **555.99** | Supported. Compatible with CUDA 11.8 through CUDA 12.5. |
| **Max CUDA Supported** | **CUDA 12.5** | Supported. |
| **Operating System** | **Windows 11 (Build 26300)** | Supported. |

---

## 2. Storage & Drive Partition Audit

| Drive | Total Storage | Used Storage | Free Storage | Status |
| :---: | :---: | :---: | :---: | :---: |
| **C:** | 224.42 GB | 192.80 GB | 31.63 GB | System Drive |
| **D:** | 250.29 GB | 33.17 GB | 217.12 GB | Available |
| **E:** | 465.76 GB | 209.36 GB | **256.39 GB** | **Primary Project & Data Drive (Dedicated)** |
| **F:** | 465.74 GB | 315.86 GB | 149.88 GB | Available |

> **Audit Finding:** The dedicated project drive (`E:`) has **256.39 GB free**, which exceeds the required 50 GB threshold by over 5x. Storage is ample for extracting required dataset subsets, checkpoints, and logs.

---

## 3. Python & Deep Learning Stack Audit

| Software / Library | Global Detected Version | Target Recommendation | Status / Action Item |
| :--- | :--- | :--- | :--- |
| **Python** | 3.13.0 (default), 3.11.9 (installed) | **Python 3.11** | Master prompt specifies Python 3.11. Python 3.11 is already installed at `AppData\Local\Programs\Python\Python311`. Recommend creating a dedicated virtual environment using Python 3.11 (`py -3.11 -m venv venv`). |
| **PyTorch** | 2.12.0+cpu | **PyTorch with CUDA 12.1 / 12.4** | Global environment currently has CPU-only PyTorch. We need PyTorch with CUDA support to utilize the RTX 2050 GPU. |
| **OpenCV** | 4.10.0 | OpenCV 4.x | Installed and functional. |
| **Streamlit** | 1.51.0 | Streamlit 1.x | Installed and functional. |
| **Scikit-Learn** | 1.8.0 | Scikit-Learn | Installed and functional. |
| **Git** | Installed | Git 2.x | Configured and connected to remote repository. |

---

## 4. Hardware Optimization Strategy for RTX 2050 (4 GB VRAM)

1. **Precision:** Always use PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast()`) to reduce VRAM consumption by ~40-50% and accelerate Tensor Cores.
2. **Batch Sizing:** Use batch sizes of `2` to `4` for segmentation and detection models.
3. **Gradient Accumulation:** Accumulate gradients over 2 to 4 steps to simulate effective batch sizes of 8 to 16 without increasing VRAM usage.
4. **Resolution Scaling:** Resize raw 1280×720 frames to 512×288 or 640×360 during pilot training.
5. **DataLoader:** Pin memory (`pin_memory=True`), use 2 worker threads (`num_workers=2`) to avoid Windows multiprocessing overhead while keeping memory stable.
