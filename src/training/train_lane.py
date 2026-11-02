"""
Phase 3 Experiment Script: Comparative Lane Detection Training and Benchmarking.
Trains Model A1 (Custom CNN) vs Model A2 (Custom U-Net) on the RTX 2050 GPU.
Profiles parameters, physical VRAM consumption, inference FPS, and validation IoU.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import time
import cv2
import numpy as np
import torch

from src.data.dataset_loaders import get_tusimple_dataloaders
from src.models.lane_detection.custom_cnn import CustomCNNSegmentor
from src.models.lane_detection.custom_unet import CustomUNet
from src.training.trainer import SegmentationTrainer

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_inference(model: torch.nn.Module,
                        device: torch.device,
                        input_shape=(1, 3, 288, 512),
                        iterations=100) -> float:
    """Measures steady-state inference latency and frames per second (FPS) on GPU."""
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup GPU
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    fps = iterations / total_time
    return fps

def save_visual_prediction(model: torch.nn.Module,
                           val_loader,
                           device: torch.device,
                           save_path: str,
                           model_name: str):
    """Generates visual overlay comparing Image, Ground Truth Mask, and Model Prediction."""
    model.eval()
    with torch.no_grad():
        images, masks, info = next(iter(val_loader))
        images_dev = images.to(device)
        logits = model(images_dev)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    # Convert first sample to visualization
    img_np = images[0].permute(1, 2, 0).cpu().numpy()
    # Denormalize ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_raw = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)

    gt_mask = (masks[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    pred_mask = (preds[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    # Create 3-panel visualization: [Input Image | Ground Truth Mask | Model Prediction]
    h, w = img_bgr.shape[:2]

    # Ground truth overlay (green)
    gt_overlay = img_bgr.copy()
    gt_color = np.zeros_like(img_bgr)
    gt_color[gt_mask > 0] = [0, 255, 0]
    gt_vis = cv2.addWeighted(gt_overlay, 0.7, gt_color, 0.3, 0)
    cv2.putText(gt_vis, "Ground Truth", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Prediction overlay (cyan)
    pred_overlay = img_bgr.copy()
    pred_color = np.zeros_like(img_bgr)
    pred_color[pred_mask > 0] = [255, 255, 0] # Cyan in BGR
    pred_vis = cv2.addWeighted(pred_overlay, 0.7, pred_color, 0.3, 0)
    cv2.putText(pred_vis, f"{model_name} Prediction", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Input image with label
    input_vis = img_bgr.copy()
    cv2.putText(input_vis, "Input Camera", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    combined = np.hstack([input_vis, gt_vis, pred_vis])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, combined)
    print(f"Saved visual prediction comparison to {save_path}")

def run_experiments(epochs: int = 5):
    split_file = 'datasets/splits/tusimple_pilot_splits.json'
    pilot_dir = 'datasets/extracted/tusimple_pilot'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Target Hardware: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load data (use num_workers=0 for drvfs/WSL2 host filesystem stability)
    train_loader, val_loader = get_tusimple_dataloaders(
        split_file=split_file,
        pilot_dir=pilot_dir,
        batch_size=4,
        target_size=(512, 288),
        num_workers=0
    )

    results = {}

    # =========================================================================
    # Experiment 1: Model A1 — Custom CNN Lane Segmentor
    # =========================================================================
    model_a1 = CustomCNNSegmentor(in_channels=3, num_classes=1).to(device)
    params_a1 = count_parameters(model_a1)
    fps_a1 = benchmark_inference(model_a1, device)

    trainer_a1 = SegmentationTrainer(
        model=model_a1,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_name='model_a1_custom_cnn'
    )
    history_a1 = trainer_a1.fit(epochs=epochs)
    final_metrics_a1 = trainer_a1.evaluate()
    save_visual_prediction(model_a1, val_loader, device,
                           'reports/visual_samples/model_a1_prediction.jpg', 'Model A1 (CNN)')

    results['Model A1 (Custom CNN)'] = {
        'parameters': params_a1,
        'inference_fps': round(fps_a1, 1),
        'best_val_iou': round(trainer_a1.best_val_iou, 4),
        'final_val_iou': round(final_metrics_a1['iou'], 4),
        'final_val_dice': round(final_metrics_a1['dice'], 4),
        'precision': round(final_metrics_a1['precision'], 4),
        'recall': round(final_metrics_a1['recall'], 4)
    }

    # =========================================================================
    # Experiment 2: Model A2 — Custom U-Net (With Skip Connections)
    # =========================================================================
    model_a2 = CustomUNet(in_channels=3, num_classes=1).to(device)
    params_a2 = count_parameters(model_a2)
    fps_a2 = benchmark_inference(model_a2, device)

    trainer_a2 = SegmentationTrainer(
        model=model_a2,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_name='model_a2_custom_unet'
    )
    history_a2 = trainer_a2.fit(epochs=epochs)
    final_metrics_a2 = trainer_a2.evaluate()
    save_visual_prediction(model_a2, val_loader, device,
                           'reports/visual_samples/model_a2_prediction.jpg', 'Model A2 (U-Net)')

    results['Model A2 (Custom U-Net)'] = {
        'parameters': params_a2,
        'inference_fps': round(fps_a2, 1),
        'best_val_iou': round(trainer_a2.best_val_iou, 4),
        'final_val_iou': round(final_metrics_a2['iou'], 4),
        'final_val_dice': round(final_metrics_a2['dice'], 4),
        'precision': round(final_metrics_a2['precision'], 4),
        'recall': round(final_metrics_a2['recall'], 4)
    }

    # Save benchmark summary
    summary_path = 'reports/lane_benchmark_comparison.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("PHASE 3: COMPARATIVE EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"  - Parameters:     {res['parameters']:,}")
        print(f"  - Inference FPS:  {res['inference_fps']} frames/sec")
        print(f"  - Best Val IoU:   {res['best_val_iou']:.4f}")
        print(f"  - Val Dice / F1:  {res['final_val_dice']:.4f}")
        print(f"  - Precision:      {res['precision']:.4f}")
        print(f"  - Recall:         {res['recall']:.4f}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    run_experiments(epochs=5)
