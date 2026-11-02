"""
Phase 4 Training Script: Module B — Custom Multi-Scale Vehicle Detector.
Trains CustomVehicleDetector on RTX 2050 using Automatic Mixed Precision (AMP).
Detects 4 vehicle classes: Car, Bus, Truck, Motorcycle.
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
import torch.nn.functional as F

from src.data.bdd_loaders import get_bdd_dataloaders
from src.models.detection.custom_detector import CustomVehicleDetector

VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle']
COLORS = [(0, 255, 0), (255, 100, 0), (200, 200, 0), (0, 165, 255)]

def compute_detector_loss(preds, obj_targets, cls_targets, box_targets):
    """
    Multi-task loss for dense anchor-free vehicle detection:
    L = L_obj + L_cls + L_box
    """
    obj_preds = preds['obj_logits']
    cls_preds = preds['cls_logits']
    box_preds = preds['bbox_preds']

    # 1. Objectness Loss (Weighted BCE with pos_weight=25 to balance 1:230 vehicle-to-background ratio)
    pos_weight = torch.tensor([25.0], device=obj_preds.device)
    obj_loss = F.binary_cross_entropy_with_logits(obj_preds, obj_targets, pos_weight=pos_weight)

    # 2. Mask for positive object cells
    pos_mask = (obj_targets > 0.5) # [B, 1, H_grid, W_grid]
    num_pos = pos_mask.sum().clamp(min=1.0)

    # 3. Classification Loss (BCE only on cells containing a vehicle)
    pos_mask_cls = pos_mask.expand_as(cls_preds)
    cls_loss = F.binary_cross_entropy_with_logits(cls_preds[pos_mask_cls], cls_targets[pos_mask_cls]) if pos_mask.sum() > 0 else torch.tensor(0.0, device=obj_preds.device)

    # 4. Box Regression Loss (Smooth L1 only on cells containing a vehicle)
    pos_mask_box = pos_mask.expand_as(box_preds)
    box_loss = F.smooth_l1_loss(box_preds[pos_mask_box], box_targets[pos_mask_box]) if pos_mask.sum() > 0 else torch.tensor(0.0, device=obj_preds.device)

    total_loss = (2.0 * obj_loss) + (1.0 * cls_loss) + (10.0 * box_loss)
    return total_loss, obj_loss.item(), cls_loss.item(), box_loss.item()

def decode_predictions(preds, conf_thresh=0.45, nms_thresh=0.40, stride=8, img_size=(512, 288)):
    """Decodes grid outputs into bounding boxes and applies NMS to remove redundant boxes."""
    from src.models.detection.nms import nms

    obj_prob = torch.sigmoid(preds['obj_logits'][0, 0]) # [H, W]
    cls_prob = torch.softmax(preds['cls_logits'][0], dim=0) # [4, H, W]
    boxes = preds['bbox_preds'][0] # [4, H, W]

    target_w, target_h = img_size
    candidate_boxes = []
    candidate_scores = []
    candidate_cats = []

    pos_y, pos_x = torch.where(obj_prob > conf_thresh)
    for y, x in zip(pos_y, pos_x):
        score = obj_prob[y, x].item()
        cat_id = torch.argmax(cls_prob[:, y, x]).item()
        cat_score = cls_prob[cat_id, y, x].item()
        final_conf = score * cat_score

        if final_conf > conf_thresh:
            cx_n, cy_n, bw_n, bh_n = boxes[:, y, x]
            cx, cy = cx_n.item() * target_w, cy_n.item() * target_h
            bw, bh = bw_n.item() * target_w, bh_n.item() * target_h
            x1 = max(0, int(cx - bw / 2.0))
            y1 = max(0, int(cy - bh / 2.0))
            x2 = min(target_w, int(cx + bw / 2.0))
            y2 = min(target_h, int(cy + bh / 2.0))
            if x2 > x1 and y2 > y1:
                candidate_boxes.append([x1, y1, x2, y2])
                candidate_scores.append(final_conf)
                candidate_cats.append(cat_id)

    if not candidate_boxes:
        return []

    boxes_arr = np.array(candidate_boxes, dtype=np.float32)
    scores_arr = np.array(candidate_scores, dtype=np.float32)

    # Per-class NMS
    final_dets = []
    for cid in range(4):
        cls_indices = [i for i, c in enumerate(candidate_cats) if c == cid]
        if not cls_indices:
            continue
        c_boxes = boxes_arr[cls_indices]
        c_scores = scores_arr[cls_indices]
        keep = nms(c_boxes, c_scores, iou_threshold=nms_thresh)
        for k in keep:
            orig_idx = cls_indices[k]
            final_dets.append((*candidate_boxes[orig_idx], candidate_scores[orig_idx], cid))

    return final_dets

def save_detector_prediction(model, val_loader, device, save_path):
    model.eval()
    with torch.no_grad():
        imgs, obj_tgt, cls_tgt, box_tgt = next(iter(val_loader))
        imgs_dev = imgs.to(device)
        preds = model(imgs_dev)

    # Denormalize
    img_np = imgs[0].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_raw = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)

    # Decode predictions
    single_pred = {
        'obj_logits': preds['obj_logits'][0:1],
        'cls_logits': preds['cls_logits'][0:1],
        'bbox_preds': preds['bbox_preds'][0:1]
    }
    detections = decode_predictions(single_pred, conf_thresh=0.25)

    vis = img_bgr.copy()
    for x1, y1, x2, y2, conf, cat_id in detections:
        cat_name = VEHICLE_CLASSES[cat_id]
        color = COLORS[cat_id]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{cat_name} {conf:.2f}", (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(vis, f"Vehicle Detections ({len(detections)} vehicles)", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)
    print(f"Visual detection overlay saved to {save_path}")

def train_detector(epochs=5, batch_size=4, lr=1e-3):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"MODULE B: TRAINING CUSTOM MULTI-SCALE VEHICLE DETECTOR")
    print(f"Target Hardware: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"{'='*70}")

    train_loader, val_loader = get_bdd_dataloaders(
        task='detection',
        split_file='datasets/splits/bdd_pilot_splits.json',
        root_dir='datasets/extracted/bdd_pilot',
        batch_size=batch_size,
        target_size=(512, 288),
        num_workers=0
    )

    model = CustomVehicleDetector(num_classes=4).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CustomVehicleDetector Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for imgs, obj_tgt, cls_tgt, box_tgt in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            obj_tgt = obj_tgt.to(device, non_blocking=True)
            cls_tgt = cls_tgt.to(device, non_blocking=True)
            box_tgt = box_tgt.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(imgs)
                loss, _, _, _ = compute_detector_loss(preds, obj_tgt, cls_tgt, box_tgt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_obj, val_cls, val_box = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, obj_tgt, cls_tgt, box_tgt in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                obj_tgt = obj_tgt.to(device, non_blocking=True)
                cls_tgt = cls_tgt.to(device, non_blocking=True)
                box_tgt = box_tgt.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    preds = model(imgs)
                    l, lo, lc, lb = compute_detector_loss(preds, obj_tgt, cls_tgt, box_tgt)

                val_loss += l.item()
                val_obj += lo
                val_cls += lc
                val_box += lb

        val_loss /= len(val_loader)
        val_obj /= len(val_loader)
        val_cls /= len(val_loader)
        val_box /= len(val_loader)

        elapsed = time.time() - t0
        vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2) if device.type == 'cuda' else 0.0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'vehicle_detector_best.pt'))

        best_flag = " [*BEST]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} (Obj: {val_obj:.3f}, Cls: {val_cls:.3f}, Box: {val_box:.3f}) | "
              f"VRAM: {vram_mb:.1f}MB | "
              f"Time: {elapsed:.2f}s{best_flag}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s! Best Val Loss: {best_val_loss:.4f}")

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
    save_detector_prediction(model, val_loader, device, 'reports/visual_samples/vehicle_detection_prediction.jpg')

if __name__ == '__main__':
    train_detector(epochs=5, batch_size=4)
