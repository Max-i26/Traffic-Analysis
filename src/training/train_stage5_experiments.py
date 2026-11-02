"""
Stage 5 Controlled Improvement Experiments Runner.
Trains and evaluates:
- v2.1: Stride-Normalized Regression (Original 200 data, Hard NMS)
- v2.2: Stride-Normalized Regression (Expanded 450 data, Hard NMS)
- v2.3: Stride-Normalized Regression (Expanded 450 data, Centerness-aware Soft-NMS)
Evaluated on the exact same 100% frozen held-out validation set.
"""

import os
import sys
import time
import json
import cv2
import argparse
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.detection.custom_fcos_detector import CustomFCOSDetector
from src.models.detection.fcos_assigner import FCOSAssigner
from src.models.detection.fcos_losses import FCOSLoss
from src.models.detection.fcos_decoder import FCOSDecoder
from src.data.fcos_loader import get_fcos_dataloaders, CLASS_NAMES
from src.evaluation.coco_detection_evaluator import COCOEvaluator

EXPERIMENT_CONFIGS = {
    'v2.1': {
        'desc': 'Stride-Normalized Regression (Original Pilot Data, Hard NMS)',
        'split_file': 'datasets/splits/bdd_pilot_splits.json',
        'normalize_by_stride': True,
        'centerness_weighted': False,
        'soft_nms': False,
        'ckpt_name': 'fcos_v2_1_stridenorm.pt'
    },
    'v2.2': {
        'desc': 'Stride-Normalized Regression + Expanded Minority Data (Hard NMS)',
        'split_file': 'datasets/splits/bdd_expanded_splits.json',
        'normalize_by_stride': True,
        'centerness_weighted': False,
        'soft_nms': False,
        'ckpt_name': 'fcos_v2_2_expanded.pt'
    },
    'v2.3': {
        'desc': 'Stride-Normalized Regression + Expanded Data + Centerness-Aware Soft-NMS',
        'split_file': 'datasets/splits/bdd_expanded_splits.json',
        'normalize_by_stride': True,
        'centerness_weighted': True,
        'soft_nms': True,
        'ckpt_name': 'fcos_v2_3_final.pt'
    }
}

def train_and_eval_experiment(exp_name: str, epochs: int = 8, batch_size: int = 4, lr: float = 3e-4):
    cfg = EXPERIMENT_CONFIGS[exp_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 75)
    print(f"RUNNING STAGE 5 EXPERIMENT: [{exp_name.upper()}]")
    print(f"Description: {cfg['desc']}")
    print(f"Hardware:    {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Dataset:     {cfg['split_file']}")
    print(f"Epochs:      {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 75)

    # 1. Model & Components
    model = CustomFCOSDetector(
        num_classes=len(CLASS_NAMES),
        fpn_channels=96,
        tower_channels=96,
        dropout=0.10,
        normalize_by_stride=cfg['normalize_by_stride']
    ).to(device)

    assigner = FCOSAssigner(
        strides=[4, 8, 16],
        scale_ranges=[(0.0, 64.0), (64.0, 128.0), (128.0, 1e7)],
        center_radius=1.5,
        normalize_by_stride=cfg['normalize_by_stride']
    )

    criterion = FCOSLoss(num_classes=len(CLASS_NAMES), alpha=[0.25, 0.50, 0.50, 0.75], gamma=2.0)

    decoder = FCOSDecoder(
        strides=[4, 8, 16],
        score_thresh=0.01,
        pre_nms_top_k=500,
        post_nms_top_k=100,
        nms_iou_thresh=0.50,
        normalize_by_stride=cfg['normalize_by_stride'],
        centerness_weighted=cfg['centerness_weighted'],
        soft_nms=cfg['soft_nms']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # 2. Data Loaders
    root_dir = 'datasets/extracted/bdd_pilot'
    train_loader, val_loader = get_fcos_dataloaders(
        split_file=cfg['split_file'],
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=0,
        use_class_aware_sampling=True
    )

    best_val_loss = float('inf')
    ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, cfg['ckpt_name'])

    # Pre-generate points
    dummy_input = torch.zeros(1, 3, 288, 512, device=device)
    with torch.no_grad():
        dummy_preds = model(dummy_input)
    feat_shapes = [(cls.size(2), cls.size(3)) for cls in dummy_preds['cls_logits']]
    points_per_level = assigner.generate_points(feat_shapes, device=device)
    all_points = torch.cat(points_per_level, dim=0)

    # Strides tensor for loss calculation
    strides_per_point = []
    for s, pts in zip(model.fpn_strides, points_per_level):
        strides_per_point.append(torch.full((pts.size(0),), s, dtype=torch.float32, device=device))
    all_strides = torch.cat(strides_per_point, dim=0)

    # 3. Training Loop
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        cls_loss_accum = 0.0
        reg_loss_accum = 0.0
        cnt_loss_accum = 0.0
        batches = 0

        for imgs, targets in train_loader:
            B = imgs.size(0)
            imgs = imgs.to(device)

            optimizer.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                preds = model(imgs)

                cls_flat = torch.cat(
                    [cls.permute(0, 2, 3, 1).reshape(B, -1, len(CLASS_NAMES)) for cls in preds['cls_logits']],
                    dim=1
                ).reshape(-1, len(CLASS_NAMES))

                reg_flat = torch.cat(
                    [reg.permute(0, 2, 3, 1).reshape(B, -1, 4) for reg in preds['bbox_preds']],
                    dim=1
                ).reshape(-1, 4)

                cnt_flat = torch.cat(
                    [cnt.reshape(B, -1) for cnt in preds['cnt_logits']],
                    dim=1
                ).reshape(-1)

                batch_cls_t = []
                batch_reg_t = []
                batch_cnt_t = []
                batch_pos_m = []

                for b in range(B):
                    gt_b = targets[b]['boxes'].to(device)
                    gt_l = targets[b]['labels'].to(device)
                    cls_t, reg_t, cnt_t, pos_m = assigner.assign_targets_single_image(
                        points_per_level, gt_b, gt_l
                    )
                    batch_cls_t.append(cls_t)
                    batch_reg_t.append(reg_t)
                    batch_cnt_t.append(cnt_t)
                    batch_pos_m.append(pos_m)

                batch_cls_t = torch.cat(batch_cls_t, dim=0)
                batch_reg_t = torch.cat(batch_reg_t, dim=0)
                batch_cnt_t = torch.cat(batch_cnt_t, dim=0)
                batch_pos_m = torch.cat(batch_pos_m, dim=0)
                batch_points = all_points.repeat(B, 1)
                batch_strides = all_strides.repeat(B)

                losses = criterion(
                    cls_flat, reg_flat, cnt_flat, batch_points,
                    batch_cls_t, batch_reg_t, batch_cnt_t, batch_pos_m,
                    strides_flat=batch_strides if cfg['normalize_by_stride'] else None
                )
                loss = losses['total_loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()
            cls_loss_accum += losses['loss_cls'].item()
            reg_loss_accum += losses['loss_reg'].item()
            cnt_loss_accum += losses['loss_cnt'].item()
            batches += 1

        scheduler.step()

        # Validation Loop
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                B = imgs.size(0)
                imgs = imgs.to(device)

                with autocast(enabled=(device.type == 'cuda')):
                    preds = model(imgs)

                    cls_flat = torch.cat(
                        [cls.permute(0, 2, 3, 1).reshape(B, -1, len(CLASS_NAMES)) for cls in preds['cls_logits']],
                        dim=1
                    ).reshape(-1, len(CLASS_NAMES))

                    reg_flat = torch.cat(
                        [reg.permute(0, 2, 3, 1).reshape(B, -1, 4) for reg in preds['bbox_preds']],
                        dim=1
                    ).reshape(-1, 4)

                    cnt_flat = torch.cat(
                        [cnt.reshape(B, -1) for cnt in preds['cnt_logits']],
                        dim=1
                    ).reshape(-1)

                    batch_cls_t = []
                    batch_reg_t = []
                    batch_cnt_t = []
                    batch_pos_m = []

                    for b in range(B):
                        gt_b = targets[b]['boxes'].to(device)
                        gt_l = targets[b]['labels'].to(device)
                        cls_t, reg_t, cnt_t, pos_m = assigner.assign_targets_single_image(
                            points_per_level, gt_b, gt_l
                        )
                        batch_cls_t.append(cls_t)
                        batch_reg_t.append(reg_t)
                        batch_cnt_t.append(cnt_t)
                        batch_pos_m.append(pos_m)

                    batch_cls_t = torch.cat(batch_cls_t, dim=0)
                    batch_reg_t = torch.cat(batch_reg_t, dim=0)
                    batch_cnt_t = torch.cat(batch_cnt_t, dim=0)
                    batch_pos_m = torch.cat(batch_pos_m, dim=0)
                    batch_points = all_points.repeat(B, 1)
                    batch_strides = all_strides.repeat(B)

                    losses = criterion(
                        cls_flat, reg_flat, cnt_flat, batch_points,
                        batch_cls_t, batch_reg_t, batch_cnt_t, batch_pos_m,
                        strides_flat=batch_strides if cfg['normalize_by_stride'] else None
                    )
                    val_loss_accum += losses['total_loss'].item()
                    val_batches += 1

        avg_train = train_loss_accum / max(1, batches)
        avg_cls = cls_loss_accum / max(1, batches)
        avg_reg = reg_loss_accum / max(1, batches)
        avg_cnt = cnt_loss_accum / max(1, batches)
        avg_val = val_loss_accum / max(1, val_batches)

        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_ckpt_path)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {avg_train:.4f} (Cls: {avg_cls:.4f}, Reg: {avg_reg:.4f}, Cnt: {avg_cnt:.4f}) | "
              f"Val Loss: {avg_val:.4f} | "
              f"{'[*] BEST SAVED' if is_best else ''}")

    train_time = time.time() - t_start
    print(f"\n[{exp_name}] Training finished in {train_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # =========================================================================
    # RELOAD BEST CHECKPOINT & COCO EVALUATION
    # =========================================================================
    print(f"Reloading best weights from: {best_ckpt_path}")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    model.eval()

    all_predictions = []
    all_ground_truths = []
    img_counter = 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            with autocast(enabled=(device.type == 'cuda')):
                preds = model(imgs)

            decoded_batch = decoder.decode_predictions(preds, score_thresh=0.01, pre_nms_top_k=500)

            for b in range(len(decoded_batch)):
                all_predictions.append({
                    'image_id': img_counter,
                    'boxes': decoded_batch[b]['boxes'],
                    'scores': decoded_batch[b]['scores'],
                    'labels': decoded_batch[b]['labels']
                })
                all_ground_truths.append({
                    'image_id': img_counter,
                    'boxes': targets[b]['boxes'].cpu().numpy(),
                    'labels': targets[b]['labels'].cpu().numpy()
                })
                img_counter += 1

    evaluator = COCOEvaluator(class_names=CLASS_NAMES)
    eval_results = evaluator.evaluate(all_predictions, all_ground_truths, operational_conf_thresh=0.35)

    # Compute Small / Medium / Large Object AP50
    area_splits = {
        'AP50_small (<32^2)': (0, 32 * 32),
        'AP50_medium (32^2-96^2)': (32 * 32, 96 * 96),
        'AP50_large (>96^2)': (96 * 96, 1e9)
    }
    scale_metrics = {}
    for scale_name, (min_a, max_a) in area_splits.items():
        scale_preds = []
        scale_gts = []
        for p, g in zip(all_predictions, all_ground_truths):
            # Filter GTs by area
            g_boxes = g['boxes']
            g_labels = g['labels']
            if len(g_boxes) > 0:
                areas = (g_boxes[:, 2] - g_boxes[:, 0]) * (g_boxes[:, 3] - g_boxes[:, 1])
                mask_g = (areas >= min_a) & (areas < max_a)
                scale_gts.append({'image_id': g['image_id'], 'boxes': g_boxes[mask_g], 'labels': g_labels[mask_g]})
            else:
                scale_gts.append(g)

            # Keep all predictions for proper recall calculation
            scale_preds.append(p)

        sc_eval = evaluator.evaluate(scale_preds, scale_gts, operational_conf_thresh=0.35)
        scale_metrics[scale_name] = sc_eval['mAP50']

    eval_results['scale_ap50'] = scale_metrics

    # =========================================================================
    # PIPELINE BENCHMARK (100 iterations)
    # =========================================================================
    sample_raw = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for _ in range(10):
        t_img = cv2.resize(sample_raw, (512, 288))
        t_img = (t_img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        t_tensor = torch.from_numpy(t_img).unsqueeze(0).to(device)
        with torch.no_grad():
            p = model(t_tensor)
            _ = decoder.decode_predictions(p, score_thresh=0.05, pre_nms_top_k=500)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t_pre_list, t_inf_list, t_dec_list = [], [], []
    for _ in range(100):
        t0 = time.perf_counter()
        t_img = cv2.resize(sample_raw, (512, 288))
        t_img = (t_img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        t_tensor = torch.from_numpy(t_img).unsqueeze(0).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        with torch.no_grad():
            with autocast(enabled=(device.type == 'cuda')):
                p = model(t_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        _ = decoder.decode_predictions(p, score_thresh=0.05, pre_nms_top_k=500)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        t_pre_list.append((t1 - t0) * 1000.0)
        t_inf_list.append((t2 - t1) * 1000.0)
        t_dec_list.append((t3 - t2) * 1000.0)

    mean_total = np.mean(t_pre_list) + np.mean(t_inf_list) + np.mean(t_dec_list)
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0

    eval_results['benchmark'] = {
        'e2e_latency_ms': round(float(mean_total), 2),
        'e2e_fps': round(1000.0 / mean_total, 1),
        'gpu_forward_ms': round(float(np.mean(t_inf_list)), 2),
        'peak_vram_mb': round(float(peak_vram), 2)
    }

    print(f"\n--- Results for [{exp_name}] ---")
    print(f"  mAP@50:     {eval_results['mAP50']:.4f}")
    print(f"  mAP@75:     {eval_results['mAP75']:.4f}")
    print(f"  mAP@50:95:  {eval_results['mAP50_95']:.4f}")
    print(f"  Car AP@50:  {eval_results['per_class']['car']['AP50']:.4f}")
    print(f"  Truck AP50: {eval_results['per_class']['truck']['AP50']:.4f}")
    print(f"  Bus AP@50:  {eval_results['per_class']['bus']['AP50']:.4f}")
    print(f"  Moto AP@50: {eval_results['per_class']['motorcycle']['AP50']:.4f}")
    print(f"  Pipeline:   {eval_results['benchmark']['e2e_fps']} FPS (VRAM: {eval_results['benchmark']['peak_vram_mb']} MB)")

    out_file = f"reports/stage5_{exp_name.replace('.', '_')}_results.json"
    with open(out_file, 'w') as f:
        json.dump(eval_results, f, indent=2)

    return eval_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='v2.1', choices=['v2.1', 'v2.2', 'v2.3', 'all'])
    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()

    if args.exp == 'all':
        for e in ['v2.1', 'v2.2', 'v2.3']:
            train_and_eval_experiment(e, epochs=args.epochs)
    else:
        train_and_eval_experiment(args.exp, epochs=args.epochs)
