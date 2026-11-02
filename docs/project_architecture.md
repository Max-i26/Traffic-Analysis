# Project Architecture & Modular Design

**Project:** Traffic Intelligence AI  
**Repository:** [https://github.com/Max-i26/Traffic-Analysis](https://github.com/Max-i26/Traffic-Analysis)

---

## 1. Directory Structure

```
Traffic-Analysis/
│
├── configs/                          # Experiment configuration YAMLs
│   ├── base_config.yaml              # Global parameters, paths, seeds
│   ├── lane_detection.yaml           # Module A: CNN, U-Net, coordinate configs
│   ├── vehicle_detection.yaml        # Module B: Detector configs
│   ├── road_segmentation.yaml        # Module C: Drivable area configs
│   ├── tracking.yaml                 # Module D: ByteTrack configs
│   └── temporal.yaml                 # Module E: LSTM & Transformer configs
│
├── datasets/                         # Data directory (Raw archives kept safe)
│   ├── raw/                          # Pointers to raw archives
│   ├── extracted/                    # Curated working subsets
│   ├── processed/                    # Normalized tensors & masks
│   ├── splits/                       # Deterministic train/val/test split manifests
│   └── dataset_registry.yaml         # Dataset audit metadata and checksums
│
├── src/                              # Core Source Code
│   ├── data/                         # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── dataset_audit.py          # Data validation script
│   │   ├── annotation_parser.py      # TuSimple & BDD parsers
│   │   ├── preprocessing.py          # Mask generator, normalization
│   │   ├── augmentation.py           # Geometry-safe augmentations
│   │   └── dataset_loaders.py        # PyTorch Dataset & DataLoader classes
│   │
│   ├── models/                       # Custom Neural Network Architectures
│   │   ├── __init__.py
│   │   ├── lane_detection/           # Module A: Custom CNN, U-Net, Coordinate Head
│   │   ├── detection/                # Module B: Custom Detector Backbone & Head
│   │   ├── segmentation/             # Module C: Drivable Area Segmentor
│   │   └── temporal/                 # Module E: CNN-LSTM & Temporal Transformer
│   │
│   ├── training/                     # Training & Optimization Engine
│   │   ├── __init__.py
│   │   ├── trainer.py                # Generic PyTorch trainer with AMP support
│   │   ├── losses.py                 # Combined Dice + BCE, SmoothL1, Focal Loss
│   │   ├── optimizer.py              # AdamW, cosine annealing scheduler
│   │   └── callbacks.py              # Early stopping, model checkpointing
│   │
│   ├── evaluation/                   # Metrics & Benchmarking
│   │   ├── __init__.py
│   │   ├── metrics.py                # IoU, Dice, mAP, FPS, latency
│   │   ├── benchmark.py              # Performance profiling on RTX 2050
│   │   └── visualization.py          # Ground truth vs prediction visual overlays
│   │
│   ├── tracking/                     # Module D: Object Tracking & Trajectories
│   │   ├── __init__.py
│   │   ├── tracker.py                # ByteTrack / Kalman Filter implementation
│   │   └── trajectory.py             # Trajectory smoothing & lane association
│   │
│   ├── inference/                    # End-to-End Inference Pipeline
│   │   ├── __init__.py
│   │   └── pipeline.py               # Frame-by-frame integrated inference
│   │
│   └── utils/                        # Shared Utilities
│       ├── __init__.py
│       ├── logger.py                 # Structured logging
│       └── config.py                 # YAML config loader
│
├── app/                              # Module F: Streamlit Application
│   ├── __init__.py
│   ├── main.py                       # Application entrypoint
│   ├── components/                   # UI components (Video player, HUD, metrics)
│   └── pages/                        # Multi-page dashboard
│
├── checkpoints/                      # Saved PyTorch model weights (.pt)
├── docs/                             # Engineering & Research Reports
│   ├── environment_report.md
│   ├── dataset_inventory.md
│   ├── dataset_quality_report.md
│   ├── development_roadmap.md
│   └── project_architecture.md
│
├── tests/                            # Automated Unit & Integration Tests
│   ├── test_annotation_parser.py
│   ├── test_dataset_loaders.py
│   └── test_models.py
│
├── .gitignore
├── README.md
└── requirements.txt
```
