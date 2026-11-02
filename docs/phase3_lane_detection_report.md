# Phase 3: Module A — Lane Detection Experiments & Hardware Benchmark

**Date:** 2026-11-02  
**Project:** Traffic Intelligence AI  
**Repository:** [https://github.com/Max-i26/Traffic-Analysis](https://github.com/Max-i26/Traffic-Analysis)  
**Execution Environment:** Ubuntu WSL2 with NVIDIA GeForce RTX 2050 (4 GB VRAM)

---

## 1. Executive Summary

Phase 3 designed, implemented, trained, and benchmarked two custom neural network architectures built from scratch:
- **Model A1:** Custom CNN Lane Segmentor (sequential encoder-decoder)
- **Model A2:** Custom U-Net (with multi-scale skip connections)

Both models were trained on the pilot split using **PyTorch Automatic Mixed Precision (AMP)** and **Combined BCE + Dice Loss**. Both models achieved real-time inference speeds (>30 FPS) on the RTX 2050 Laptop GPU.

---

## 2. Comparative Benchmark Matrix

| Metric / Specification | Model A1: Custom CNN | Model A2: Custom U-Net | Performance Difference & Research Insight |
| :--- | :---: | :---: | :--- |
| **Architecture Type** | Sequential ConvNet (no skips) | Multi-Scale U-Net (4 skip levels) | U-Net preserves high-resolution edge features |
| **Trainable Parameters** | **2,330,529** (~2.33M) | **7,763,041** (~7.76M) | U-Net has 3.3x more capacity |
| **Peak VRAM Allocated** | **615.0 MB** | **999.1 MB** | Both comfortably under the 4,096 MB VRAM ceiling |
| **Inference Throughput** | **65.0 FPS** (15.4 ms/frame) | **34.2 FPS** (29.2 ms/frame) | CNN is faster; U-Net maintains real-time (>30 FPS) |
| **Best Validation IoU** | 0.4889 | **0.5062** | **U-Net achieves +1.73% higher IoU** |
| **Validation Dice / F1** | 0.6522 | **0.6672** | **U-Net achieves higher mask overlap** |
| **Precision** | 0.5830 | **0.6123** | **U-Net generates fewer false positive pixels (+2.93%)** |
| **Recall** | 0.7474 | 0.7404 | Both models reliably capture lane continuity |

---

## 3. Key Research Insights

1. **Impact of Skip Connections:**
   - In Model A1 (sequential encoder-decoder), downsampling to $\frac{1}{16}$ spatial resolution causes fine boundary information to blur.
   - Model A2 (U-Net) concatenates shallow encoder features directly with decoder feature maps. This yields sharper lane edges, reducing false-positive lane boundaries and boosting precision from **0.5830 to 0.6123**.

2. **Mixed Precision & Hardware Efficiency on RTX 2050:**
   - Using `torch.amp.autocast('cuda')` utilizes the RTX 2050's Ampere Tensor Cores.
   - Peak VRAM footprint stayed under **1,000 MB** throughout backpropagation, confirming that full training runs with batch size 4 or 8 will not trigger CUDA Out-of-Memory (OOM).

3. **Loss Function Synergy:**
   - Because lane pixels constitute only $\sim 3-5\%$ of traffic scenes, standard Cross-Entropy alone tends to collapse towards background prediction.
   - Combining $\text{BCEWithLogitsLoss}$ with $\text{DiceLoss}$ effectively guided gradient descent towards high-overlap lane segmentation from Epoch 1.

---

## 4. Visual Verification Artifacts

Visual prediction overlays comparing camera input, ground-truth labels, and model predictions:
- Model A1 Predictions: [`reports/visual_samples/model_a1_prediction.jpg`](file:///E:/Neural_Network_project/reports/visual_samples/model_a1_prediction.jpg)
- Model A2 Predictions: [`reports/visual_samples/model_a2_prediction.jpg`](file:///E:/Neural_Network_project/reports/visual_samples/model_a2_prediction.jpg)
- Saved Checkpoints:
  - `checkpoints/model_a1_custom_cnn_best.pt`
  - `checkpoints/model_a2_custom_unet_best.pt`
- Numerical Log: `reports/lane_benchmark_comparison.json`

---

## 5. Next Step: Phase 4 — Vehicle Detection & Road Segmentation (Modules B & C)

With lane detection models established, the next phase covers:
1. **Module C:** Drivable area road surface segmentation using BDD100K drivable maps.
2. **Module B:** Custom multi-scale vehicle detector backbone for cars, buses, trucks, and motorcycles using our standardized detection labels.
3. Comparative training and mIoU / mAP benchmarking on the RTX 2050.
