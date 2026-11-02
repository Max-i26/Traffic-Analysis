"""
Exhaustive Localization & Ranking Audit:
Investigates the complete IoU distribution, confidence ranking,
missed ground truths, and false positives at both 0.50 and 0.75 matching thresholds
for the frozen primary baseline (fcos_v2_2_expanded.pt).
"""

import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.detection.custom_fcos_detector import CustomFCOSDetector
from src.models.detection.fcos_decoder import FCOSDecoder
from src.data.fcos_loader import get_fcos_dataloaders, CLASS_NAMES
from src.evaluation.coco_detection_evaluator import box_iou_matrix, COCOEvaluator

def run_iou_distribution_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = 'checkpoints/fcos_v2_2_expanded.pt'
    split_file = 'datasets/splits/bdd_pilot_splits.json'
    root_dir = 'datasets/extracted/bdd_pilot'

    model = CustomFCOSDetector(
        num_classes=len(CLASS_NAMES),
        fpn_channels=96,
        tower_channels=96,
        normalize_by_stride=True
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()

    decoder = FCOSDecoder(
        strides=[4, 8, 16],
        score_thresh=0.005,
        pre_nms_top_k=500,
        post_nms_top_k=100,
        nms_iou_thresh=0.50,
        normalize_by_stride=True
    )

    _, val_loader = get_fcos_dataloaders(
        split_file=split_file,
        root_dir=root_dir,
        batch_size=4,
        num_workers=0,
        use_class_aware_sampling=False
    )

    all_preds = []
    all_gts = []
    img_counter = 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            B = imgs.size(0)
            imgs = imgs.to(device)
            preds = model(imgs)
            decoded_batch = decoder.decode_predictions(preds, score_thresh=0.005, pre_nms_top_k=500)

            for b in range(B):
                all_preds.append({
                    'image_id': img_counter,
                    'boxes': decoded_batch[b]['boxes'],
                    'scores': decoded_batch[b]['scores'],
                    'labels': decoded_batch[b]['labels']
                })
                all_gts.append({
                    'image_id': img_counter,
                    'boxes': targets[b]['boxes'].cpu().numpy(),
                    'labels': targets[b]['labels'].cpu().numpy()
                })
                img_counter += 1

    # 1. Evaluate standard COCO at 0.50 and 0.75
    evaluator = COCOEvaluator(class_names=CLASS_NAMES)
    eval_metrics = evaluator.evaluate(all_preds, all_gts, operational_conf_thresh=0.35)

    # 2. Analyze Full IoU Distribution for Ground Truth Objects
    gt_best_ious = []
    gt_best_scores = []
    gt_covered_at_50 = 0
    gt_covered_at_75 = 0
    total_gts = sum(len(g['boxes']) for g in all_gts)

    iou_bins = {
        '[0.00, 0.25)': 0,
        '[0.25, 0.50)': 0,
        '[0.50, 0.60)': 0,
        '[0.60, 0.70)': 0,
        '[0.70, 0.75)': 0,
        '[0.75, 0.85)': 0,
        '[0.85, 1.00]': 0
    }

    for p, g in zip(all_preds, all_gts):
        p_boxes = p['boxes']
        p_scores = p['scores']
        p_labels = p['labels']
        g_boxes = g['boxes']
        g_labels = g['labels']

        if len(g_boxes) == 0:
            continue

        if len(p_boxes) == 0:
            for _ in range(len(g_boxes)):
                gt_best_ious.append(0.0)
                iou_bins['[0.00, 0.25)'] += 1
            continue

        for gb, gl in zip(g_boxes, g_labels):
            # Same class predictions
            same_cls = np.where(p_labels == gl)[0]
            if len(same_cls) == 0:
                gt_best_ious.append(0.0)
                iou_bins['[0.00, 0.25)'] += 1
                continue

            ious = box_iou_matrix(gb[None, :], p_boxes[same_cls])[0]
            best_idx = np.argmax(ious)
            best_iou = float(ious[best_idx])
            best_score = float(p_scores[same_cls[best_idx]])

            gt_best_ious.append(best_iou)
            gt_best_scores.append(best_score)

            if best_iou >= 0.50:
                gt_covered_at_50 += 1
            if best_iou >= 0.75:
                gt_covered_at_75 += 1

            if best_iou < 0.25:
                iou_bins['[0.00, 0.25)'] += 1
            elif best_iou < 0.50:
                iou_bins['[0.25, 0.50)'] += 1
            elif best_iou < 0.60:
                iou_bins['[0.50, 0.60)'] += 1
            elif best_iou < 0.70:
                iou_bins['[0.60, 0.70)'] += 1
            elif best_iou < 0.75:
                iou_bins['[0.70, 0.75)'] += 1
            elif best_iou < 0.85:
                iou_bins['[0.75, 0.85)'] += 1
            else:
                iou_bins['[0.85, 1.00]'] += 1

    # 3. Match Count & False Positive Breakdown at 0.50 vs 0.75
    tp_count_50 = 0
    fp_count_50 = 0
    tp_count_75 = 0
    fp_count_75 = 0

    for p, g in zip(all_preds, all_gts):
        p_boxes = p['boxes']
        p_scores = p['scores']
        p_labels = p['labels']
        g_boxes = g['boxes']
        g_labels = g['labels']

        # Consider detections with score >= 0.25
        mask = p_scores >= 0.25
        p_boxes = p_boxes[mask]
        p_labels = p_labels[mask]

        if len(p_boxes) == 0:
            continue

        if len(g_boxes) == 0:
            fp_count_50 += len(p_boxes)
            fp_count_75 += len(p_boxes)
            continue

        # IoU 0.50 matching
        g_matched_50 = np.zeros(len(g_boxes), dtype=bool)
        for pb, pl in zip(p_boxes, p_labels):
            valid = np.where((g_labels == pl) & (~g_matched_50))[0]
            if len(valid) == 0:
                fp_count_50 += 1
                continue
            ious = box_iou_matrix(pb[None, :], g_boxes[valid])[0]
            b_idx = np.argmax(ious)
            if ious[b_idx] >= 0.50:
                tp_count_50 += 1
                g_matched_50[valid[b_idx]] = True
            else:
                fp_count_50 += 1

        # IoU 0.75 matching
        g_matched_75 = np.zeros(len(g_boxes), dtype=bool)
        for pb, pl in zip(p_boxes, p_labels):
            valid = np.where((g_labels == pl) & (~g_matched_75))[0]
            if len(valid) == 0:
                fp_count_75 += 1
                continue
            ious = box_iou_matrix(pb[None, :], g_boxes[valid])[0]
            b_idx = np.argmax(ious)
            if ious[b_idx] >= 0.75:
                tp_count_75 += 1
                g_matched_75[valid[b_idx]] = True
            else:
                fp_count_75 += 1

    audit_summary = {
        'total_ground_truth_objects': total_gts,
        'gt_covered_at_iou_50': gt_covered_at_50,
        'gt_covered_at_iou_75': gt_covered_at_75,
        'gt_coverage_rate_at_50_pct': round((gt_covered_at_50 / max(1, total_gts)) * 100.0, 2),
        'gt_coverage_rate_at_75_pct': round((gt_covered_at_75 / max(1, total_gts)) * 100.0, 2),
        'iou_distribution_bins': {
            k: {'count': v, 'pct_of_all_gt': round((v / max(1, total_gts)) * 100.0, 2)}
            for k, v in iou_bins.items()
        },
        'matching_at_operating_thresh_0_25': {
            'threshold_0_50': {'TP': tp_count_50, 'FP': fp_count_50, 'Precision': round(tp_count_50 / max(1, tp_count_50 + fp_count_50), 4)},
            'threshold_0_75': {'TP': tp_count_75, 'FP': fp_count_75, 'Precision': round(tp_count_75 / max(1, tp_count_75 + fp_count_75), 4)}
        },
        'coco_metrics': {
            'mAP50': eval_metrics['mAP50'],
            'mAP75': eval_metrics['mAP75'],
            'mAP50_95': eval_metrics['mAP50_95']
        }
    }

    out_file = 'reports/localization_iou_and_ranking_audit.json'
    with open(out_file, 'w') as f:
        json.dump(audit_summary, f, indent=2)

    print("=" * 80)
    print("      LOCALIZATION IoU DISTRIBUTION & RANKING AUDIT")
    print("=" * 80)
    print(f"Total Ground Truth Objects: {total_gts}")
    print(f"GT Objects with IoU >= 0.50 Candidate: {gt_covered_at_50} ({audit_summary['gt_coverage_rate_at_50_pct']}%)")
    print(f"GT Objects with IoU >= 0.75 Candidate: {gt_covered_at_75} ({audit_summary['gt_coverage_rate_at_75_pct']}%)")
    print("\nComplete IoU Distribution across all 546 Ground Truth Objects:")
    for b_name, b_info in audit_summary['iou_distribution_bins'].items():
        print(f"  {b_name:<16}: {b_info['count']:>4} ({b_info['pct_of_all_gt']:>5.1f}%)")
    print("\nOperating Detection Performance (at confidence >= 0.25):")
    print(f"  IoU @ 0.50: TP = {tp_count_50}, FP = {fp_count_50}, Precision = {audit_summary['matching_at_operating_thresh_0_25']['threshold_0_50']['Precision']*100:.2f}%")
    print(f"  IoU @ 0.75: TP = {tp_count_75}, FP = {fp_count_75}, Precision = {audit_summary['matching_at_operating_thresh_0_25']['threshold_0_75']['Precision']*100:.2f}%")
    print(f"  COCO mAP@50 = {eval_metrics['mAP50']:.4f} | mAP@75 = {eval_metrics['mAP75']:.4f}")
    print(f"\n[OK] Audit written to {out_file}")

    return audit_summary

if __name__ == '__main__':
    run_iou_distribution_audit()
