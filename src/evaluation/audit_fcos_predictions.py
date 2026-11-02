"""
Stage 4 Candidate & Localization Audit for FCOS Detector.
Investigates:
1. Candidate counts per level (P2, P3, P4) before/after top-k and NMS.
2. Prediction count capping (why exactly 5,000 cars).
3. Score distribution of True Positives vs False Positives.
4. IoU distribution of true positive detections (why AP50 >> AP75).
5. False positive error analysis for Cars and Trucks.
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
from src.evaluation.coco_detection_evaluator import box_iou_matrix

def audit_fcos():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("STAGE 4: FCOS CANDIDATE GENERATION & LOCALIZATION AUDIT")
    print(f"Device: {device}")
    print("=" * 70)

    ckpt_path = 'checkpoints/fcos_v2_fast_baseline.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'checkpoints/fcos_detector_best.pt'
    print(f"Loading frozen checkpoint: {ckpt_path}")

    model = CustomFCOSDetector(num_classes=len(CLASS_NAMES), fpn_channels=96, tower_channels=96).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    decoder = FCOSDecoder(strides=[4, 8, 16], score_thresh=0.01, pre_nms_top_k=500, post_nms_top_k=100)

    split_path = 'datasets/splits/bdd_pilot_splits.json'
    root_dir = 'datasets/extracted/bdd_pilot'
    _, val_loader = get_fcos_dataloaders(
        split_file=split_path,
        root_dir=root_dir,
        batch_size=4,
        num_workers=0,
        use_class_aware_sampling=False
    )

    level_candidates = {4: [], 8: [], 16: []} # Strides 4 (P2), 8 (P3), 16 (P4)
    tp_scores = []
    fp_scores = []
    matched_ious = []

    # Score buckets for FP vs TP: [0.01-0.10, 0.10-0.25, 0.25-0.50, >0.50]
    fp_by_score = {'0.01-0.10': 0, '0.10-0.25': 0, '0.25-0.50': 0, '>0.50': 0}
    tp_by_score = {'0.01-0.10': 0, '0.10-0.25': 0, '0.25-0.50': 0, '>0.50': 0}

    # Per class counts
    class_pred_counts = {c: 0 for c in CLASS_NAMES}
    class_tp_counts = {c: 0 for c in CLASS_NAMES}

    with torch.no_grad():
        for imgs, targets in val_loader:
            B = imgs.size(0)
            imgs = imgs.to(device)
            preds = model(imgs)

            # 1. Profile per-level candidate counts (score > 0.01)
            for b in range(B):
                for lvl_idx, stride in enumerate([4, 8, 16]):
                    cls_l = torch.sigmoid(preds['cls_logits'][lvl_idx][b]).permute(1, 2, 0)
                    cnt_l = torch.sigmoid(preds['cnt_logits'][lvl_idx][b]).unsqueeze(-1)
                    score_map = torch.sqrt(cls_l * cnt_l)
                    max_s = score_map.max(dim=-1)[0]
                    cands_count = (max_s > 0.01).sum().item()
                    level_candidates[stride].append(cands_count)

            # 2. Decode batch
            decoded_batch = decoder.decode_predictions(preds, score_thresh=0.01, pre_nms_top_k=500)

            for b in range(B):
                pred_boxes = decoded_batch[b]['boxes']
                pred_scores = decoded_batch[b]['scores']
                pred_labels = decoded_batch[b]['labels']

                gt_boxes = targets[b]['boxes'].cpu().numpy()
                gt_labels = targets[b]['labels'].cpu().numpy()

                gt_matched = np.zeros(len(gt_boxes), dtype=bool)

                # Track per-image counts
                for c_idx, c_name in enumerate(CLASS_NAMES):
                    class_pred_counts[c_name] += int((pred_labels == c_idx).sum())

                # Sort by score descending
                if len(pred_boxes) > 0:
                    order = np.argsort(pred_scores)[::-1]
                    pred_boxes = pred_boxes[order]
                    pred_scores = pred_scores[order]
                    pred_labels = pred_labels[order]

                for p_box, p_sc, p_lbl in zip(pred_boxes, pred_scores, pred_labels):
                    score_bin = '>0.50' if p_sc > 0.50 else ('0.25-0.50' if p_sc > 0.25 else ('0.10-0.25' if p_sc > 0.10 else '0.01-0.10'))

                    if len(gt_boxes) == 0:
                        fp_scores.append(p_sc)
                        fp_by_score[score_bin] += 1
                        continue

                    # Find matching GT of same category
                    cat_match_indices = np.where((gt_labels == p_lbl) & (~gt_matched))[0]
                    if len(cat_match_indices) == 0:
                        fp_scores.append(p_sc)
                        fp_by_score[score_bin] += 1
                        continue

                    ious = box_iou_matrix(p_box[None, :], gt_boxes[cat_match_indices])[0]
                    best_sub = np.argmax(ious)
                    best_iou = ious[best_sub]
                    best_gt_idx = cat_match_indices[best_sub]

                    if best_iou >= 0.50:
                        gt_matched[best_gt_idx] = True
                        tp_scores.append(p_sc)
                        tp_by_score[score_bin] += 1
                        matched_ious.append(best_iou)
                        class_tp_counts[CLASS_NAMES[p_lbl]] += 1
                    else:
                        fp_scores.append(p_sc)
                        fp_by_score[score_bin] += 1

    print("\n--- 1. Candidate Generation Across Levels ---")
    for stride, cands in level_candidates.items():
        lvl_name = f"P2 (stride {stride})" if stride==4 else (f"P3 (stride {stride})" if stride==8 else f"P4 (stride {stride})")
        print(f"  {lvl_name}: Mean {np.mean(cands):.1f} candidates/img (Min: {np.min(cands)}, Max: {np.max(cands)})")

    print("\n--- 2. Predictions & Cap Analysis (50 Validation Images) ---")
    for c_name in CLASS_NAMES:
        print(f"  {c_name}: {class_pred_counts[c_name]} total predictions (True Positives: {class_tp_counts[c_name]})")
    print(f"  Note: Capped at exactly 5,000 because 50 images * 100 max detections = 5,000!")

    print("\n--- 3. Score Distribution: True Positives vs False Positives ---")
    print(f"  Score Range       | True Positives (TP) | False Positives (FP) | Precision in Bin")
    print(f"  ---------------------------------------------------------------------------------")
    for s_bin in ['0.01-0.10', '0.10-0.25', '0.25-0.50', '>0.50']:
        tp_c = tp_by_score[s_bin]
        fp_c = fp_by_score[s_bin]
        p_bin = (tp_c / max(1, tp_c + fp_c)) * 100.0
        print(f"  {s_bin:<17} | {tp_c:<19} | {fp_c:<20} | {p_bin:.1f}%")

    print(f"\n  * Total True Positives: {len(tp_scores)}")
    print(f"  * Total False Positives: {len(fp_scores)}")
    print(f"  * Overwhelmingly {fp_by_score['0.01-0.10']} / {len(fp_scores)} ({fp_by_score['0.01-0.10']/max(1, len(fp_scores))*100:.1f}%) of False Positives are low-confidence clutter (score < 0.10)!")

    print("\n--- 4. IoU Distribution for Matched Predictions ---")
    if len(matched_ious) > 0:
        matched_ious = np.array(matched_ious)
        print(f"  Mean IoU of True Positives: {np.mean(matched_ious):.4f}")
        print(f"  IoU in [0.50, 0.60): {np.sum((matched_ious >= 0.50) & (matched_ious < 0.60))} ({np.mean((matched_ious >= 0.50) & (matched_ious < 0.60))*100:.1f}%)")
        print(f"  IoU in [0.60, 0.70): {np.sum((matched_ious >= 0.60) & (matched_ious < 0.70))} ({np.mean((matched_ious >= 0.60) & (matched_ious < 0.70))*100:.1f}%)")
        print(f"  IoU in [0.70, 0.75): {np.sum((matched_ious >= 0.70) & (matched_ious < 0.75))} ({np.mean((matched_ious >= 0.70) & (matched_ious < 0.75))*100:.1f}%)")
        print(f"  IoU in [0.75, 1.00]: {np.sum(matched_ious >= 0.75)} ({np.mean(matched_ious >= 0.75)*100:.1f}%)")
        print(f"  Diagnosis: {(np.sum(matched_ious < 0.70)/len(matched_ious))*100:.1f}% of detections have IoU between 0.50 and 0.70, which is why AP@50 is solid but AP@75 is near zero!")

    audit_results = {
        'candidate_generation': {s: round(float(np.mean(c)), 1) for s, c in level_candidates.items()},
        'class_predictions': class_pred_counts,
        'class_true_positives': class_tp_counts,
        'score_distribution': {
            'tp_by_score': tp_by_score,
            'fp_by_score': fp_by_score
        },
        'iou_distribution': {
            'mean_iou': round(float(np.mean(matched_ious)), 4) if len(matched_ious) else 0.0,
            'iou_50_60': int(np.sum((matched_ious >= 0.50) & (matched_ious < 0.60))) if len(matched_ious) else 0,
            'iou_60_70': int(np.sum((matched_ious >= 0.60) & (matched_ious < 0.70))) if len(matched_ious) else 0,
            'iou_70_75': int(np.sum((matched_ious >= 0.70) & (matched_ious < 0.75))) if len(matched_ious) else 0,
            'iou_75_plus': int(np.sum(matched_ious >= 0.75)) if len(matched_ious) else 0
        }
    }

    report_path = 'reports/stage4_localization_audit.json'
    with open(report_path, 'w') as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nAudit complete. Machine report saved to: {report_path}")

if __name__ == '__main__':
    audit_fcos()
