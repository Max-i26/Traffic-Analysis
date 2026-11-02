"""
Per-Class Operational Confidence Threshold Sweep for fcos_v2_2_expanded.pt.
Finds the optimal operating threshold (tau) for Car, Truck, Bus, Motorcycle individually.
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

def run_per_class_sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print("STAGE 6: PER-CLASS CONFIDENCE THRESHOLD & OPERATING POINT SWEEP")
    print("=" * 75)

    ckpt_path = 'checkpoints/fcos_v2_2_expanded.pt'
    model = CustomFCOSDetector(
        num_classes=len(CLASS_NAMES),
        fpn_channels=96,
        tower_channels=96,
        normalize_by_stride=True
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    decoder = FCOSDecoder(
        strides=[4, 8, 16],
        score_thresh=0.01,
        pre_nms_top_k=500,
        post_nms_top_k=100,
        nms_iou_thresh=0.50,
        normalize_by_stride=True
    )

    split_path = 'datasets/splits/bdd_pilot_splits.json'
    root_dir = 'datasets/extracted/bdd_pilot'
    _, val_loader = get_fcos_dataloaders(
        split_file=split_path,
        root_dir=root_dir,
        batch_size=4,
        num_workers=0,
        use_class_aware_sampling=False
    )

    all_decoded = []
    all_gts = []

    with torch.no_grad():
        for imgs, targets in val_loader:
            B = imgs.size(0)
            imgs = imgs.to(device)
            preds = model(imgs)
            decoded_batch = decoder.decode_predictions(preds, score_thresh=0.01, pre_nms_top_k=500)

            for b in range(B):
                all_decoded.append(decoded_batch[b])
                all_gts.append({
                    'boxes': targets[b]['boxes'].cpu().numpy(),
                    'labels': targets[b]['labels'].cpu().numpy()
                })

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    class_sweep_summary = {}

    for cat_id, cat_name in enumerate(CLASS_NAMES):
        print(f"\n--- Category: [{cat_name.upper()}] ---")
        print(f"{'Thresh':<8} | {'TP':<6} | {'FP':<8} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
        print("-" * 60)

        n_gt_cat = sum(int((gt['labels'] == cat_id).sum()) for gt in all_gts)
        cat_records = []
        best_f1 = -1.0
        best_tau = 0.35

        for thresh in thresholds:
            tp, fp = 0, 0
            for pred, gt in zip(all_decoded, all_gts):
                p_mask = (pred['labels'] == cat_id) & (pred['scores'] >= thresh)
                g_mask = (gt['labels'] == cat_id)

                p_boxes = pred['boxes'][p_mask]
                p_scores = pred['scores'][p_mask]
                g_boxes = gt['boxes'][g_mask]

                if len(p_boxes) == 0:
                    continue

                order = np.argsort(p_scores)[::-1]
                p_boxes = p_boxes[order]

                g_matched = np.zeros(len(g_boxes), dtype=bool)

                for pb in p_boxes:
                    if len(g_boxes) == 0:
                        fp += 1
                        continue

                    ious = box_iou_matrix(pb[None, :], g_boxes)[0]
                    best_match = np.argmax(ious)
                    if ious[best_match] >= 0.50 and not g_matched[best_match]:
                        tp += 1
                        g_matched[best_match] = True
                    else:
                        fp += 1

            prec = tp / max(1e-7, tp + fp)
            rec = tp / max(1e-7, n_gt_cat)
            f1 = 2 * (prec * rec) / max(1e-7, prec + rec)

            if f1 > best_f1 and tp > 0:
                best_f1 = f1
                best_tau = thresh

            cat_records.append({
                'threshold': thresh,
                'tp': tp,
                'fp': fp,
                'precision': round(float(prec), 4),
                'recall': round(float(rec), 4),
                'f1': round(float(f1), 4)
            })

            print(f"{thresh:<8.2f} | {tp:<6} | {fp:<8} | {prec*100:<9.2f}% | {rec*100:<9.2f}% | {f1*100:<9.2f}%")

        class_sweep_summary[cat_name] = {
            'gt_count': n_gt_cat,
            'best_operating_threshold': best_tau,
            'best_f1_score': round(float(best_f1), 4) if best_f1 > 0 else 0.0,
            'sweep': cat_records
        }

    out_file = 'reports/stage6_per_class_threshold_sweep.json'
    with open(out_file, 'w') as f:
        json.dump(class_sweep_summary, f, indent=2)
    print(f"\nSaved per-class sweep to: {out_file}")

if __name__ == '__main__':
    run_per_class_sweep()
