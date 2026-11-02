"""
Module B v2 Training Script: Custom FCOS Multi-Scale Vehicle Detector.
Trains CustomFCOSDetector on BDD100K pilot dataset with:
- Multi-scale FPN (P2, P3, P4)
- Scale-aware target routing ([0,64], [64,128], [128,inf])
- FCOS center sampling and minimal-area overlap resolution
- Focal Loss + Centerness BCE + GIoU Loss
- FP16 Automatic Mixed Precision (AMP) optimized for RTX 2050 (4 GB VRAM)
- Mandatory best checkpoint reloading before final evaluation and benchmarking.
"""

import os
import sys
import time
import json
import cv2
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

def train_fcos(epochs: int = 10,
               batch_size: int = 4,
               lr: float = 5e-4,
               device_str: str = 'cuda'):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("MODULE B v2: TRAINING CUSTOM FCOS MULTI-SCALE VEHICLE DETECTOR")
    print(f"Target Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Base LR: {lr}")
    print("=" * 70)

    # 1. Model & Components
    model = CustomFCOSDetector(num_classes=len(CLASS_NAMES)).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CustomFCOSDetector Trainable Parameters: {num_params:,}")

    assigner = FCOSAssigner(
        strides=[4, 8, 16],
        scale_ranges=[(0.0, 64.0), (64.0, 128.0), (128.0, 1e7)],
        center_radius=1.5
    )
    # Class-weighted focal loss: gives higher gradients to minority classes
    criterion = FCOSLoss(num_classes=len(CLASS_NAMES), alpha=[0.25, 0.50, 0.50, 0.75], gamma=2.0)
    decoder = FCOSDecoder(
        strides=[4, 8, 16],
        score_thresh=0.05,
        pre_nms_top_k=500,
        post_nms_top_k=100,
        nms_iou_thresh=0.50
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # 2. Data Loaders
    split_path = 'datasets/splits/bdd_pilot_splits.json'
    root_dir = 'datasets/extracted/bdd_pilot'
    train_loader, val_loader = get_fcos_dataloaders(
        split_file=split_path,
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=0,
        use_class_aware_sampling=True
    )

    best_val_loss = float('inf')
    ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, 'fcos_detector_best.pt')

    # Pre-generate points for assigner
    dummy_input = torch.zeros(1, 3, 288, 512, device=device)
    with torch.no_grad():
        dummy_preds = model(dummy_input)
    feat_shapes = [(cls.size(2), cls.size(3)) for cls in dummy_preds['cls_logits']]
    points_per_level = assigner.generate_points(feat_shapes, device=device)
    all_points = torch.cat(points_per_level, dim=0) # [Total_Points, 2]

    # 3. Training Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        cls_loss_accum = 0.0
        reg_loss_accum = 0.0
        cnt_loss_accum = 0.0
        pos_points_accum = 0
        batches = 0

        for imgs, targets in train_loader:
            B = imgs.size(0)
            imgs = imgs.to(device)

            optimizer.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                preds = model(imgs)

                # Flatten predictions across batch and feature points
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

                # Batch assign targets
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

                losses = criterion(
                    cls_flat, reg_flat, cnt_flat, batch_points,
                    batch_cls_t, batch_reg_t, batch_cnt_t, batch_pos_m
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
            pos_points_accum += losses['num_pos']
            batches += 1

        scheduler.step()

        # Validation Step
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

                    losses = criterion(
                        cls_flat, reg_flat, cnt_flat, batch_points,
                        batch_cls_t, batch_reg_t, batch_cnt_t, batch_pos_m
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

    total_training_sec = time.time() - start_time
    print(f"\nTraining completed in {total_training_sec:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # =========================================================================
    # MANDATORY STEP (ISSUE 5 FIX): RELOAD BEST CHECKPOINT
    # =========================================================================
    print(f"\nReloading best checkpoint from {best_ckpt_path} for final evaluation...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # =========================================================================
    # RIGOROUS COCO EVALUATION (ISSUES 6 & 7 FIX)
    # =========================================================================
    print("\nRunning COCO Evaluation on Unseen Validation Set...")
    all_predictions = []
    all_ground_truths = []
    img_counter = 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            with autocast(enabled=(device.type == 'cuda')):
                preds = model(imgs)

            # Efficient ranking evaluation with pre-nms top-k to prevent candidate explosion
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

    print("\n" + "=" * 70)
    print("MODULE B v2 (CUSTOM FCOS DETECTOR) EVALUATION RESULTS:")
    print("=" * 70)
    print(json.dumps(eval_results, indent=2))

    report_path = 'reports/fcos_detector_evaluation.json'
    with open(report_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved evaluation metrics to: {report_path}")

    # =========================================================================
    # VISUAL PREDICTION OVERLAY
    # =========================================================================
    val_vis_img = None
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            with autocast(enabled=(device.type == 'cuda')):
                preds = model(imgs)
            vis_preds = decoder.decode_predictions(preds, score_thresh=0.20)

            # Denormalize image 0 for visualization
            img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_vis = ((img_np * std + mean) * 255.0).clip(0, 255).astype(np.uint8)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

            palette = {
                0: (0, 255, 0),     # Car: Green
                1: (255, 200, 0),   # Bus: Cyan/Yellow
                2: (0, 165, 255),   # Truck: Orange
                3: (0, 0, 255)      # Motorcycle: Red
            }

            for box, score, lbl in zip(vis_preds[0]['boxes'], vis_preds[0]['scores'], vis_preds[0]['labels']):
                x1, y1, x2, y2 = map(int, box)
                color = palette.get(lbl, (255, 255, 255))
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                cat_name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else str(lbl)
                text = f"{cat_name} {score:.2f}"
                cv2.putText(img_vis, text, (x1, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            val_vis_img = img_vis
            break

    if val_vis_img is not None:
        vis_path = 'reports/visual_samples/fcos_detector_prediction.jpg'
        cv2.imwrite(vis_path, val_vis_img)
        print(f"Saved visual prediction sample: {vis_path}")

    # =========================================================================
    # ISSUE 11 FIX: COMPLETE PIPELINE BENCHMARK (PREPROCESS + INFERENCE + NMS)
    # =========================================================================
    print("\nRunning Verified Complete Pipeline Benchmark (100 iterations)...")
    sample_raw = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        t_img = cv2.resize(sample_raw, (512, 288))
        t_img = (t_img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        t_tensor = torch.from_numpy(t_img).unsqueeze(0).to(device)
        with torch.no_grad():
            p = model(t_tensor)
            _ = decoder.decode_predictions(p, score_thresh=0.05)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t_pre_list = []
    t_inf_list = []
    t_dec_list = []

    for _ in range(100):
        # 1. Preprocessing
        t0 = time.perf_counter()
        t_img = cv2.resize(sample_raw, (512, 288))
        t_img = (t_img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        t_tensor = torch.from_numpy(t_img).unsqueeze(0).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # 2. GPU Model Forward
        with torch.no_grad():
            with autocast(enabled=(device.type == 'cuda')):
                p = model(t_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # 3. Complete Decoding & Class-wise NMS
        _ = decoder.decode_predictions(p, score_thresh=0.05)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        t_pre_list.append((t1 - t0) * 1000.0)
        t_inf_list.append((t2 - t1) * 1000.0)
        t_dec_list.append((t3 - t2) * 1000.0)

    mean_pre = np.mean(t_pre_list)
    mean_inf = np.mean(t_inf_list)
    mean_dec = np.mean(t_dec_list)
    mean_total = mean_pre + mean_inf + mean_dec
    e2e_fps = 1000.0 / mean_total
    gpu_fps = 1000.0 / mean_inf
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0

    benchmark_summary = {
        'resolution': '512x288',
        'preprocessing_ms': round(float(mean_pre), 2),
        'gpu_forward_ms': round(float(mean_inf), 2),
        'decoding_and_nms_ms': round(float(mean_dec), 2),
        'total_e2e_latency_ms': round(float(mean_total), 2),
        'pure_gpu_fps': round(float(gpu_fps), 1),
        'complete_pipeline_fps': round(float(e2e_fps), 1),
        'peak_gpu_vram_mb': round(float(peak_vram), 2)
    }

    print("\nComplete Pipeline Benchmark Summary:")
    print(json.dumps(benchmark_summary, indent=2))

    eval_results['complete_pipeline_benchmark'] = benchmark_summary
    with open(report_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    train_fcos(epochs=epochs, batch_size=4, lr=5e-4)
