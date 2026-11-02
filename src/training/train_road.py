"""
Phase 4 Training Script: Module C — Custom Drivable Road Surface Segmentation.
Trains RoadSegmentor on RTX 2050 using Automatic Mixed Precision (AMP).
"""

import os
import sys

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import cv2
import numpy as np
import torch
import torch.nn as nn

from src.data.bdd_loaders import get_bdd_dataloaders
from src.models.segmentation.road_segmentor import RoadSegmentor
from src.training.losses import BCEDiceLoss
from src.evaluation.metrics import compute_batch_metrics

def save_road_prediction(model, val_loader, device, save_path):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images_dev = images.to(device)
        logits = model(images_dev)
        preds = (torch.sigmoid(logits) > 0.5).float()

    # Denormalize
    img_np = images[0].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_raw = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)

    gt_mask = (masks[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    pred_mask = (preds[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    # Overlays
    gt_vis = img_bgr.copy()
    gt_color = np.zeros_like(img_bgr)
    gt_color[gt_mask > 0] = [0, 255, 0] # Green ground truth drivable area
    gt_vis = cv2.addWeighted(gt_vis, 0.65, gt_color, 0.35, 0)
    cv2.putText(gt_vis, "Ground Truth Drivable Area", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    pred_vis = img_bgr.copy()
    pred_color = np.zeros_like(img_bgr)
    pred_color[pred_mask > 0] = [255, 140, 0] # Blue/Orange drivable area prediction
    pred_vis = cv2.addWeighted(pred_vis, 0.65, pred_color, 0.35, 0)
    cv2.putText(pred_vis, "RoadSegmentor (ASPP) Pred", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)

    input_vis = img_bgr.copy()
    cv2.putText(input_vis, "Input Camera", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    combined = np.hstack([input_vis, gt_vis, pred_vis])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, combined)
    print(f"Visual prediction saved to {save_path}")

def train_road(epochs=5, batch_size=4, lr=1e-3):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"MODULE C: TRAINING DRIVABLE ROAD SURFACE SEGMENTOR")
    print(f"Target Hardware: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"{'='*70}")

    train_loader, val_loader = get_bdd_dataloaders(
        task='drivable',
        split_file='datasets/splits/bdd_pilot_splits.json',
        root_dir='datasets/extracted/bdd_pilot',
        batch_size=batch_size,
        target_size=(512, 288),
        num_workers=0
    )

    model = RoadSegmentor(in_channels=3, num_classes=1).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RoadSegmentor Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=1.0)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_iou = 0.0
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        metrics_sum = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                val_loss += loss.item()
                bm = compute_batch_metrics(logits, masks)
                for k in metrics_sum:
                    metrics_sum[k] += bm[k]

        val_loss /= len(val_loader)
        val_iou = metrics_sum['iou'] / len(val_loader)
        val_dice = metrics_sum['dice'] / len(val_loader)
        val_prec = metrics_sum['precision'] / len(val_loader)
        val_rec = metrics_sum['recall'] / len(val_loader)

        elapsed = time.time() - t0
        vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2) if device.type == 'cuda' else 0.0

        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'road_segmentor_best.pt'))

        best_flag = " [*BEST]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val mIoU: {val_iou:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"VRAM: {vram_mb:.1f}MB | "
              f"Time: {elapsed:.2f}s{best_flag}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s! Best Val mIoU: {best_val_iou:.4f}")

    # Benchmark FPS
    model.eval()
    dummy = torch.randn(1, 3, 288, 512, device=device)
    for _ in range(15): _ = model(dummy)
    if device.type == 'cuda': torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(100): _ = model(dummy)
    if device.type == 'cuda': torch.cuda.synchronize()
    fps = 100.0 / (time.time() - t_start)
    print(f"Inference Speed on RTX 2050: {fps:.1f} FPS")

    # Visual prediction
    save_road_prediction(model, val_loader, device, 'reports/visual_samples/road_segmentation_prediction.jpg')

if __name__ == '__main__':
    train_road(epochs=5, batch_size=4)
