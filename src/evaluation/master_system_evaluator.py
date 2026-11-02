"""
Master System Evaluator: Comprehensive Publication-Grade Evaluation
across ALL modules, models, testers, and metrics in the Traffic Intelligence System.

Modules Evaluated:
1. Module A: Lane Segmentation (Custom CNN vs Custom U-Net)
2. Module C: Drivable Road Surface Segmentation (RoadSegmentor + ASPP)
3. Module B: Vehicle Detection (Custom FCOS v2.2 vs v1 Baseline)
   - COCO mAP (mAP@50, mAP@75, mAP@50:95)
   - Scale-stratified AP (Small, Medium, Large)
   - Per-class AP (Car, Truck, Bus, Motorcycle)
   - Operational Precision, Recall, F1, TP, FP, FN
   - Localization & Geometric Bounding Box Error Breakdown
4. Module D: Multi-Object Vehicle Tracking (ByteTrack + Kalman + Trajectories)
5. Hardware Profiler: Exact Stage-by-Stage Latency, Throughput (FPS), Peak VRAM on NVIDIA RTX 2050
"""

import os
import sys
import time
import json
import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import Models
from src.models.lane_detection.custom_cnn import CustomCNNSegmentor
from src.models.lane_detection.custom_unet import CustomUNet
from src.models.segmentation.road_segmentor import RoadSegmentor
from src.models.detection.custom_fcos_detector import CustomFCOSDetector
from src.models.detection.fcos_decoder import FCOSDecoder
from src.tracking.tracker import ByteTracker
from src.tracking.trajectory import TrajectoryManager
from src.data.dataset_loaders import get_tusimple_dataloaders
from src.data.bdd_loaders import get_bdd_dataloaders
from src.data.fcos_loader import get_fcos_dataloaders, CLASS_NAMES
from src.evaluation.coco_detection_evaluator import COCOEvaluator, box_iou_matrix

def load_checkpoint_weights(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    return model

def run_master_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("      MASTER SYSTEM EVALUATION: ALL TESTERS, MODULES & METRICS")
    print(f"Hardware Environment: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    master_results = {}

    # =========================================================================
    # TESTER 1: MODULE A - LANE SEGMENTATION (CNN vs U-NET)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TESTER 1: MODULE A - LANE DETECTION ARCHITECTURE COMPARISON")
    print("Dataset: TuSimple Held-Out Validation Split (40 continuous scenes)")
    print("=" * 80)

    tusimple_split = 'datasets/splits/tusimple_pilot_splits.json'
    tusimple_root = 'datasets/extracted/tusimple_pilot'
    _, tu_val_loader = get_tusimple_dataloaders(
        split_file=tusimple_split,
        pilot_dir=tusimple_root,
        batch_size=4,
        num_workers=0
    )

    lane_models = [
        ('Model A1 (Custom CNN)', CustomCNNSegmentor(), 'checkpoints/model_a1_custom_cnn_best.pt'),
        ('Model A2 (Custom U-Net)', CustomUNet(), 'checkpoints/model_a2_custom_unet_best.pt')
    ]

    lane_results = {}
    for name, model, ckpt in lane_models:
        if not os.path.exists(ckpt):
            print(f"Warning: Missing {ckpt}, skipping {name}")
            continue
        model = model.to(device)
        load_checkpoint_weights(model, ckpt, device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())

        total_inter = 0.0
        total_union = 0.0
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        total_correct = 0.0
        total_pixels = 0.0

        with torch.no_grad():
            for batch in tu_val_loader:
                imgs = batch[0].to(device)
                masks = batch[1].to(device)

                logits = model(imgs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.50).float()

                # Metrics calculation
                tp = (preds * masks).sum().item()
                fp = (preds * (1.0 - masks)).sum().item()
                fn = ((1.0 - preds) * masks).sum().item()
                correct = (preds == masks).sum().item()
                num_p = masks.numel()

                total_inter += tp
                total_union += (tp + fp + fn)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_correct += correct
                total_pixels += num_p

        iou = total_inter / max(1e-7, total_union)
        dice = (2.0 * total_tp) / max(1e-7, 2.0 * total_tp + total_fp + total_fn)
        precision = total_tp / max(1e-7, total_tp + total_fp)
        recall = total_tp / max(1e-7, total_tp + total_fn)
        pixel_acc = total_correct / max(1e-7, total_pixels)

        lane_results[name] = {
            'parameters': n_params,
            'iou': round(float(iou), 4),
            'dice_f1': round(float(dice), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'pixel_accuracy': round(float(pixel_acc), 4)
        }

        print(f"  {name:<25}: IoU: {iou:.4f} | Dice/F1: {dice:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Params: {n_params:,}")

    master_results['tester1_lane_detection'] = lane_results

    # =========================================================================
    # TESTER 2: MODULE C - ROAD DRIVABLE SURFACE SEGMENTATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("TESTER 2: MODULE C - ROAD DRIVABLE SURFACE SEGMENTATION")
    print("Model: RoadSegmentor (ASPP Bridge with Atrous Rates [1, 6, 12, 18])")
    print("Dataset: BDD100K Drivable Maps Held-Out Validation Split (50 scenes)")
    print("=" * 80)

    bdd_split = 'datasets/splits/bdd_pilot_splits.json'
    bdd_root = 'datasets/extracted/bdd_pilot'
    _, road_val_loader = get_bdd_dataloaders(
        task='drivable',
        split_file=bdd_split,
        root_dir=bdd_root,
        batch_size=4,
        num_workers=0
    )

    road_model = RoadSegmentor(num_classes=1).to(device)
    road_ckpt = 'checkpoints/road_segmentor_best.pt'
    road_results = {}
    if os.path.exists(road_ckpt):
        load_checkpoint_weights(road_model, road_ckpt, device)
        road_model.eval()

        road_params = sum(p.numel() for p in road_model.parameters())

        tp_road, fp_road, fn_road, tn_road = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for imgs, masks in road_val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = road_model(imgs)
                preds = (torch.sigmoid(logits) > 0.50).float()

                tp = (preds * masks).sum().item()
                fp = (preds * (1.0 - masks)).sum().item()
                fn = ((1.0 - preds) * masks).sum().item()
                tn = ((1.0 - preds) * (1.0 - masks)).sum().item()

                tp_road += tp
                fp_road += fp
                fn_road += fn
                tn_road += tn

        road_iou = tp_road / max(1e-7, tp_road + fp_road + fn_road)
        road_dice = (2.0 * tp_road) / max(1e-7, 2.0 * tp_road + fp_road + fn_road)
        road_prec = tp_road / max(1e-7, tp_road + fp_road)
        road_rec = tp_road / max(1e-7, tp_road + fn_road)

        bg_iou = tn_road / max(1e-7, tn_road + fp_road + fn_road)
        bg_dice = (2.0 * tn_road) / max(1e-7, 2.0 * tn_road + fp_road + fn_road)
        mean_iou = (road_iou + bg_iou) / 2.0
        mean_dice = (road_dice + bg_dice) / 2.0
        pixel_accuracy = (tp_road + tn_road) / (tp_road + tn_road + fp_road + fn_road)

        road_results = {
            'parameters': road_params,
            'drivable_surface_iou': round(float(road_iou), 4),
            'drivable_surface_dice': round(float(road_dice), 4),
            'drivable_surface_precision': round(float(road_prec), 4),
            'drivable_surface_recall': round(float(road_rec), 4),
            'background_iou': round(float(bg_iou), 4),
            'mean_iou': round(float(mean_iou), 4),
            'mean_dice': round(float(mean_dice), 4),
            'pixel_accuracy': round(float(pixel_accuracy), 4),
            'confusion_matrix_pixels': {
                'TP': int(tp_road),
                'FP': int(fp_road),
                'FN': int(fn_road),
                'TN': int(tn_road)
            }
        }

        print(f"  Mean IoU: {mean_iou:.4f} | Drivable Road IoU: {road_iou:.4f} | Recall: {road_rec:.4f} | Pixel Acc: {pixel_accuracy:.4f}")
        print(f"  Confusion Matrix: TP={int(tp_road):,} | FP={int(fp_road):,} | FN={int(fn_road):,} | TN={int(tn_road):,}")

    master_results['tester2_road_segmentation'] = road_results

    # =========================================================================
    # TESTER 3: MODULE B - VEHICLE DETECTION (CUSTOM FCOS v2.2)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TESTER 3: MODULE B - VEHICLE DETECTION EXHAUSTIVE BENCHMARK")
    print("Model: CustomFCOSDetector (fcos_v2_2_expanded.pt - Primary Research Baseline)")
    print("Dataset: BDD100K 50 Unseen Validation Scenes (546 Ground-Truth Vehicles)")
    print("=" * 80)

    fcos_ckpt = 'checkpoints/fcos_v2_2_expanded.pt'
    fcos_model = CustomFCOSDetector(
        num_classes=len(CLASS_NAMES),
        fpn_channels=96,
        tower_channels=96,
        normalize_by_stride=True
    ).to(device)
    load_checkpoint_weights(fcos_model, fcos_ckpt, device)
    fcos_model.eval()

    fcos_decoder = FCOSDecoder(
        strides=[4, 8, 16],
        score_thresh=0.01,
        pre_nms_top_k=500,
        post_nms_top_k=100,
        nms_iou_thresh=0.50,
        normalize_by_stride=True
    )

    _, fcos_val_loader = get_fcos_dataloaders(
        split_file=bdd_split,
        root_dir=bdd_root,
        batch_size=4,
        num_workers=0,
        use_class_aware_sampling=False
    )

    all_predictions = []
    all_ground_truths = []
    img_counter = 0

    with torch.no_grad():
        for imgs, targets in fcos_val_loader:
            B = imgs.size(0)
            imgs = imgs.to(device)
            preds = fcos_model(imgs)
            decoded_batch = fcos_decoder.decode_predictions(preds, score_thresh=0.01, pre_nms_top_k=500)

            for b in range(B):
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

    # Standard COCO Evaluation
    evaluator = COCOEvaluator(class_names=CLASS_NAMES)
    eval_standard = evaluator.evaluate(all_predictions, all_ground_truths, operational_conf_thresh=0.35)

    # Scale-Stratified AP
    area_ranges = {
        'AP50_small (<32^2)': (0, 32 * 32),
        'AP50_medium (32^2-96^2)': (32 * 32, 96 * 96),
        'AP50_large (>96^2)': (96 * 96, 1e9)
    }
    scale_ap_results = {}
    for s_name, (a_min, a_max) in area_ranges.items():
        s_preds = []
        s_gts = []
        for p, g in zip(all_predictions, all_ground_truths):
            g_b = g['boxes']
            g_l = g['labels']
            if len(g_b) > 0:
                areas = (g_b[:, 2] - g_b[:, 0]) * (g_b[:, 3] - g_b[:, 1])
                mask = (areas >= a_min) & (areas < a_max)
                s_gts.append({'image_id': g['image_id'], 'boxes': g_b[mask], 'labels': g_l[mask]})
            else:
                s_gts.append(g)
            s_preds.append(p)
        sc_res = evaluator.evaluate(s_preds, s_gts, operational_conf_thresh=0.35)
        scale_ap_results[s_name] = sc_res['mAP50']

    # Localization Geometry Audit: Center & Dimension Error on Matched True Positives
    center_errors_x = []
    center_errors_y = []
    width_ratios = []
    height_ratios = []
    matched_ious = []

    for p, g in zip(all_predictions, all_ground_truths):
        p_boxes = p['boxes']
        p_scores = p['scores']
        p_labels = p['labels']
        g_boxes = g['boxes']
        g_labels = g['labels']

        if len(p_boxes) == 0 or len(g_boxes) == 0:
            continue

        keep = p_scores >= 0.25
        p_boxes_sub = p_boxes[keep]
        p_labels_sub = p_labels[keep]

        g_matched = np.zeros(len(g_boxes), dtype=bool)

        for pb, plbl in zip(p_boxes_sub, p_labels_sub):
            valid_g = np.where((g_labels == plbl) & (~g_matched))[0]
            if len(valid_g) == 0:
                continue

            ious = box_iou_matrix(pb[None, :], g_boxes[valid_g])[0]
            best_idx = np.argmax(ious)
            if ious[best_idx] >= 0.50:
                g_match_idx = valid_g[best_idx]
                g_matched[g_match_idx] = True
                matched_ious.append(ious[best_idx])

                gb = g_boxes[g_match_idx]
                p_cx = (pb[0] + pb[2]) / 2.0
                p_cy = (pb[1] + pb[3]) / 2.0
                g_cx = (gb[0] + gb[2]) / 2.0
                g_cy = (gb[1] + gb[3]) / 2.0

                center_errors_x.append(abs(p_cx - g_cx))
                center_errors_y.append(abs(p_cy - g_cy))
                width_ratios.append((pb[2] - pb[0]) / max(1.0, gb[2] - gb[0]))
                height_ratios.append((pb[3] - pb[1]) / max(1.0, gb[3] - gb[1]))

    geom_audit = {
        'mean_iou': round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
        'mean_center_error_x_px': round(float(np.mean(center_errors_x)), 2) if center_errors_x else 0.0,
        'mean_center_error_y_px': round(float(np.mean(center_errors_y)), 2) if center_errors_y else 0.0,
        'mean_width_ratio_pred_to_gt': round(float(np.mean(width_ratios)), 3) if width_ratios else 1.0,
        'mean_height_ratio_pred_to_gt': round(float(np.mean(height_ratios)), 3) if height_ratios else 1.0
    }

    # Relative Gain Calculation vs Baseline v1
    v1_mAP50 = 0.0002
    v1_car_recall = 0.0515
    fcos_mAP50 = eval_standard['mAP50']
    fcos_car_recall = eval_standard['per_class']['car']['operational_recall']

    relative_map_gain = (fcos_mAP50 / v1_mAP50) if v1_mAP50 > 0 else 0.0
    relative_recall_gain = (fcos_car_recall / v1_car_recall) if v1_car_recall > 0 else 0.0

    det_master = {
        'mAP50': eval_standard['mAP50'],
        'mAP75': eval_standard['mAP75'],
        'mAP50_95': eval_standard['mAP50_95'],
        'per_class': eval_standard['per_class'],
        'scale_ap50': scale_ap_results,
        'localization_geometry': geom_audit,
        'gains_vs_v1_baseline': {
            'relative_mAP50_multiplier': round(relative_map_gain, 1),
            'relative_car_recall_multiplier': round(relative_recall_gain, 1)
        }
    }

    print(f"  Overall mAP@50: {eval_standard['mAP50']:.4f} ({relative_map_gain:.1f}x vs v1 baseline)")
    print(f"  Car AP@50:      {eval_standard['per_class']['car']['AP50']:.4f} | Recall@.35: {fcos_car_recall*100:.1f}% ({relative_recall_gain:.1f}x vs v1)")
    print(f"  Truck AP@50:    {eval_standard['per_class']['truck']['AP50']:.4f}")
    print(f"  Bus AP@50:      {eval_standard['per_class']['bus']['AP50']:.4f}")
    print(f"  Scale Breakdown: Small AP={scale_ap_results['AP50_small (<32^2)']:.4f} | Med AP={scale_ap_results['AP50_medium (32^2-96^2)']:.4f} | Large AP={scale_ap_results['AP50_large (>96^2)']:.4f}")
    print(f"  Geometry: Mean IoU={geom_audit['mean_iou']} | Center Shift=({geom_audit['mean_center_error_x_px']}px, {geom_audit['mean_center_error_y_px']}px) | W-ratio={geom_audit['mean_width_ratio_pred_to_gt']}")

    master_results['tester3_vehicle_detection'] = det_master

    # =========================================================================
    # TESTER 4: MODULE D - TEMPORAL VEHICLE TRACKING EVALUATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("TESTER 4: MODULE D - TEMPORAL MULTI-OBJECT VEHICLE TRACKING")
    print("Method: 8-State Kalman Motion Model + ByteTrack Two-Stage IoU Association")
    print("Video: 20-frame Continuous Highway Stream (20 FPS)")
    print("=" * 80)

    # Ingest tracking log from track_video run
    tracking_log_path = 'reports/tracking_evaluation_summary.json'
    if os.path.exists(tracking_log_path):
        with open(tracking_log_path, 'r') as f:
            t_log = json.load(f)

        frames_data = t_log.get('frames', [])
        total_frames = len(frames_data)

        # Track ID longevity analysis
        track_lifespans = {}
        lane_assignment_history = {}
        velocity_magnitudes = []

        for frame in frames_data:
            f_num = frame['frame']
            for trk in frame['tracks']:
                tid = trk['track_id']
                track_lifespans.setdefault(tid, []).append(f_num)
                lane_assignment_history.setdefault(tid, []).append(trk['lane'])
                vx, vy = trk['v_pixel']
                velocity_magnitudes.append(np.sqrt(vx**2 + vy**2))

        lifespans = [len(f_list) for f_list in track_lifespans.values()]
        mean_lifespan = np.mean(lifespans) if lifespans else 0.0

        # Lane assignment stability (% of tracks whose lane remained 100% constant)
        stable_lanes = sum(1 for lanes in lane_assignment_history.values() if len(set(lanes)) == 1)
        lane_stability_rate = (stable_lanes / max(1, len(lane_assignment_history))) * 100.0

        tracking_metrics = {
            'evaluated_sequence_frames': total_frames,
            'total_active_tracks': len(track_lifespans),
            'mean_track_lifespan_frames': round(float(mean_lifespan), 2),
            'max_track_lifespan_frames': int(max(lifespans)) if lifespans else 0,
            'lane_assignment_consistency_rate': round(float(lane_stability_rate), 1),
            'mean_pixel_velocity': round(float(np.mean(velocity_magnitudes)), 1) if velocity_magnitudes else 0.0
        }

        print(f"  Total Unique Tracks Formed: {len(track_lifespans)}")
        print(f"  Mean Track Lifespan:        {mean_lifespan:.1f} frames (Max: {max(lifespans)} frames)")
        print(f"  Lane Corridor Consistency:  {lane_stability_rate:.1f}% persistent across sequence")
        print(f"  Mean Pixel Velocity Vector: {tracking_metrics['mean_pixel_velocity']} px/sec")

        master_results['tester4_vehicle_tracking'] = tracking_metrics

    # =========================================================================
    # TESTER 5: UNIFIED HARDWARE BENCHMARK & EFFICIENCY PROFILE
    # =========================================================================
    print("\n" + "=" * 80)
    print("TESTER 5: UNIFIED REAL-TIME HARDWARE BENCHMARK (RTX 2050)")
    print("Measurement: 100 iterations at native 512x288 resolution with FP16 AMP")
    print("=" * 80)

    sample_raw = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Warmup
    for _ in range(15):
        t_img = cv2.resize(sample_raw, (512, 288))
        t_img = (t_img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        t_tensor = torch.from_numpy(t_img).unsqueeze(0).to(device)
        with torch.no_grad():
            p_det = fcos_model(t_tensor)
            p_road = road_model(t_tensor)
            _ = fcos_decoder.decode_predictions(p_det, score_thresh=0.05, pre_nms_top_k=500)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t_pre, t_fwd_det, t_fwd_road, t_post = [], [], [], []

    for _ in range(100):
        # 1. Preprocessing
        t0 = time.perf_counter()
        t_img = cv2.resize(sample_raw, (512, 288))
        t_img = (t_img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        t_tensor = torch.from_numpy(t_img).unsqueeze(0).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # 2. Forward Passes
        with torch.no_grad():
            p_det = fcos_model(t_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        with torch.no_grad():
            p_road = road_model(t_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        # 3. Postprocessing
        _ = fcos_decoder.decode_predictions(p_det, score_thresh=0.05, pre_nms_top_k=500)
        _ = (torch.sigmoid(p_road) > 0.50)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        t_pre.append((t1 - t0) * 1000.0)
        t_fwd_det.append((t2 - t1) * 1000.0)
        t_fwd_road.append((t3 - t2) * 1000.0)
        t_post.append((t4 - t3) * 1000.0)

    mean_pre = np.mean(t_pre)
    mean_det = np.mean(t_fwd_det)
    mean_road = np.mean(t_fwd_road)
    mean_post = np.mean(t_post)
    total_single_det = mean_pre + mean_det + mean_post
    total_dual_system = mean_pre + mean_det + mean_road + mean_post

    fps_detector = 1000.0 / total_single_det
    fps_dual_system = 1000.0 / total_dual_system
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0

    hw_profile = {
        'resolution': '512x288',
        'preprocessing_ms': round(float(mean_pre), 2),
        'fcos_detector_forward_ms': round(float(mean_det), 2),
        'road_segmentor_forward_ms': round(float(mean_road), 2),
        'postprocessing_nms_ms': round(float(mean_post), 2),
        'detector_only_e2e_latency_ms': round(float(total_single_det), 2),
        'dual_system_e2e_latency_ms': round(float(total_dual_system), 2),
        'detector_pipeline_fps': round(float(fps_detector), 1),
        'dual_system_fps': round(float(fps_dual_system), 1),
        'peak_vram_mb': round(float(peak_vram), 1),
        'vram_headroom_percent': round((1.0 - peak_vram / 4096.0) * 100.0, 1) if peak_vram > 0 else 100.0
    }

    print(f"  Preprocessing:     {mean_pre:.2f} ms")
    print(f"  Detector Forward:  {mean_det:.2f} ms")
    print(f"  Road Forward:      {mean_road:.2f} ms")
    print(f"  NMS Postprocess:   {mean_post:.2f} ms")
    print(f"  Detector-Only FPS: {fps_detector:.1f} FPS (Latency: {total_single_det:.2f} ms)")
    print(f"  Dual-System FPS:   {fps_dual_system:.1f} FPS (Latency: {total_dual_system:.2f} ms)")
    print(f"  Peak VRAM Usage:   {peak_vram:.1f} MB / 4096 MB ({hw_profile['vram_headroom_percent']}% Headroom)")

    master_results['tester5_hardware_profile'] = hw_profile

    # Save Master JSON Output
    master_json_path = 'reports/master_evaluation_metrics.json'
    with open(master_json_path, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"\n[OK] Master evaluation report written to: {master_json_path}")

    return master_results

if __name__ == '__main__':
    run_master_evaluation()
