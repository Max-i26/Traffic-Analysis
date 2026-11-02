import torch
import numpy as np
from typing import Dict, List, Tuple

def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two sets of boxes [N, 4] and [M, 4] (x1, y1, x2, y2).
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    union = np.clip(union, a_min=1e-7, a_max=None)
    return inter / union

def compute_ap_101_point(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Standard COCO 101-point interpolated Average Precision.
    """
    if len(recalls) == 0:
        return 0.0
    recall_levels = np.linspace(0.0, 1.0, 101)
    prec_at_rec = np.zeros(101, dtype=np.float32)

    for i, r in enumerate(recall_levels):
        prec_candidates = precisions[recalls >= r]
        if len(prec_candidates) > 0:
            prec_at_rec[i] = np.max(prec_candidates)
        else:
            prec_at_rec[i] = 0.0
    return float(np.mean(prec_at_rec))

class COCOEvaluator:
    """
    Academic-grade COCO evaluator fixing:
    1. Zero-prediction classes are strictly counted in the mean AP denominator.
    2. Full ranking evaluation down to floor confidence (0.001), without artificial cutoff.
    3. Proper 10-threshold mAP@50:95 integration and operational metric reporting.
    """
    def __init__(self, class_names: List[str], iou_thresholds: List[float] = None):
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds or [0.50 + 0.05 * i for i in range(10)]

    def evaluate(self,
                 all_predictions: List[Dict],
                 all_ground_truths: List[Dict],
                 operational_conf_thresh: float = 0.35) -> Dict:
        """
        all_predictions: list of dicts per image:
            {
               'image_id': int,
               'boxes': np.ndarray [K, 4] (x1, y1, x2, y2),
               'scores': np.ndarray [K],
               'labels': np.ndarray [K] (int class id)
            }
        all_ground_truths: list of dicts per image:
            {
               'image_id': int,
               'boxes': np.ndarray [M, 4] (x1, y1, x2, y2),
               'labels': np.ndarray [M] (int class id)
            }
        """
        results_per_class = {}
        all_ap50 = []
        all_ap75 = []
        all_ap50_95 = []

        total_op_tp = 0
        total_op_fp = 0
        total_op_gt = 0

        # Build ground truth index by image_id
        gt_by_img = {gt['image_id']: gt for gt in all_ground_truths}
        pred_by_img = {p['image_id']: p for p in all_predictions}

        for cat_id, cat_name in enumerate(self.class_names):
            # Gather all GTs for this class
            gt_boxes_per_img = {}
            n_gt_class = 0
            for img_id, gt_data in gt_by_img.items():
                mask = (gt_data['labels'] == cat_id)
                boxes = gt_data['boxes'][mask]
                gt_boxes_per_img[img_id] = boxes
                n_gt_class += len(boxes)

            # Gather all predictions for this class
            pred_list = []
            for img_id, pred_data in pred_by_img.items():
                mask = (pred_data['labels'] == cat_id)
                boxes = pred_data['boxes'][mask]
                scores = pred_data['scores'][mask]
                for b, s in zip(boxes, scores):
                    pred_list.append({
                        'image_id': img_id,
                        'box': b,
                        'score': float(s)
                    })

            # Check if class has ground truths
            if n_gt_class == 0:
                continue # No GT instances in dataset for this class

            total_op_gt += n_gt_class

            # ISSUE 6 FIX: If class has GTs but 0 predictions, it gets AP=0.0 and is INCLUDED in mean
            if len(pred_list) == 0:
                all_ap50.append(0.0)
                all_ap75.append(0.0)
                all_ap50_95.append(0.0)
                results_per_class[cat_name] = {
                    'AP50': 0.0,
                    'AP75': 0.0,
                    'AP50_95': 0.0,
                    'GT_count': n_gt_class,
                    'Pred_count': 0,
                    'operational_precision': 0.0,
                    'operational_recall': 0.0
                }
                continue

            # Sort predictions globally by score descending
            pred_list.sort(key=lambda x: x['score'], reverse=True)

            # Evaluate across each IoU threshold
            ap_at_ious = []
            op_precision = 0.0
            op_recall = 0.0

            for iou_thresh in self.iou_thresholds:
                matched_gt = {img_id: np.zeros(len(gt_boxes_per_img[img_id]), dtype=bool)
                              for img_id in gt_boxes_per_img}

                tp = np.zeros(len(pred_list), dtype=np.float32)
                fp = np.zeros(len(pred_list), dtype=np.float32)

                for p_idx, pred in enumerate(pred_list):
                    img_id = pred['image_id']
                    p_box = pred['box'][None, :]
                    g_boxes = gt_boxes_per_img.get(img_id, np.empty((0, 4)))

                    if len(g_boxes) == 0:
                        fp[p_idx] = 1.0
                        continue

                    ious = box_iou_matrix(p_box, g_boxes)[0]
                    best_match = np.argmax(ious)
                    best_iou = ious[best_match]

                    if best_iou >= iou_thresh and not matched_gt[img_id][best_match]:
                        tp[p_idx] = 1.0
                        matched_gt[img_id][best_match] = True
                    else:
                        fp[p_idx] = 1.0

                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(fp)
                recalls = cum_tp / n_gt_class
                precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)

                ap = compute_ap_101_point(recalls, precisions)
                ap_at_ious.append(ap)

                # Compute operational precision/recall at IoU=0.50
                if abs(iou_thresh - 0.50) < 1e-4:
                    # Filter predictions with score >= operational_conf_thresh
                    op_mask = np.array([p['score'] >= operational_conf_thresh for p in pred_list])
                    if np.any(op_mask):
                        last_idx = np.where(op_mask)[0][-1]
                        op_tp_val = cum_tp[last_idx]
                        op_fp_val = cum_fp[last_idx]
                        op_precision = float(op_tp_val / max(1e-7, op_tp_val + op_fp_val))
                        op_recall = float(op_tp_val / n_gt_class)
                        total_op_tp += int(op_tp_val)
                        total_op_fp += int(op_fp_val)

            ap50 = ap_at_ious[0]
            ap75 = ap_at_ious[5] if len(ap_at_ious) > 5 else 0.0
            ap50_95 = float(np.mean(ap_at_ious))

            all_ap50.append(ap50)
            all_ap75.append(ap75)
            all_ap50_95.append(ap50_95)

            results_per_class[cat_name] = {
                'AP50': round(ap50, 4),
                'AP75': round(ap75, 4),
                'AP50_95': round(ap50_95, 4),
                'GT_count': n_gt_class,
                'Pred_count': len(pred_list),
                'operational_precision': round(op_precision, 4),
                'operational_recall': round(op_recall, 4)
            }

        mean_ap50 = float(np.mean(all_ap50)) if all_ap50 else 0.0
        mean_ap75 = float(np.mean(all_ap75)) if all_ap75 else 0.0
        mean_ap50_95 = float(np.mean(all_ap50_95)) if all_ap50_95 else 0.0

        overall_op_prec = float(total_op_tp / max(1e-7, total_op_tp + total_op_fp))
        overall_op_rec = float(total_op_tp / max(1e-7, total_op_gt))

        return {
            'mAP50': round(mean_ap50, 4),
            'mAP75': round(mean_ap75, 4),
            'mAP50_95': round(mean_ap50_95, 4),
            'evaluated_classes_count': len(all_ap50),
            'operational_threshold': operational_conf_thresh,
            'overall_operational_precision': round(overall_op_prec, 4),
            'overall_operational_recall': round(overall_op_rec, 4),
            'per_class': results_per_class
        }
