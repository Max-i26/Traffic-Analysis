import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.detection.custom_detector import CustomVehicleDetector
from src.data.bdd_loaders import get_bdd_dataloaders
from src.evaluation.coco_detection_evaluator import COCOEvaluator
from src.models.detection.nms import nms

CATEGORIES = ['car', 'bus', 'truck', 'motorcycle']

def evaluate_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Baseline Detector on: {device}")

    # Load baseline model
    model = CustomVehicleDetector(num_classes=len(CATEGORIES)).to(device)
    ckpt_path = 'checkpoints/vehicle_detector_best.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Dataloader
    split_path = 'datasets/splits/bdd_pilot_splits.json'
    root_dir = 'datasets/extracted/bdd_pilot'
    _, val_loader = get_bdd_dataloaders(
        task='detection',
        split_file=split_path,
        root_dir=root_dir,
        batch_size=4,
        num_workers=0
    )

    all_predictions = []
    all_ground_truths = []
    img_counter = 0

    eval_conf_floor = 0.005 # Low floor to let ranking evaluate PR curve properly

    with torch.no_grad():
        for imgs, obj_targets, cls_targets, box_targets in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs)

            obj_prob = torch.sigmoid(preds['obj_logits']) # [B, 1, 36, 64]
            cls_prob = torch.softmax(preds['cls_logits'], dim=1) # [B, 4, 36, 64]
            box_preds = preds['bbox_preds'] # [B, 4, 36, 64] (cx, cy, bw, bh) normalized

            B = imgs.size(0)
            for b in range(B):
                img_id = img_counter
                img_counter += 1

                # Extract Ground Truth for image
                pos_mask_gt = obj_targets[b, 0] > 0.5
                gt_boxes = []
                gt_labels = []
                if pos_mask_gt.any():
                    gt_cy_cx = box_targets[b, :2, pos_mask_gt].T.cpu().numpy()
                    gt_wh = box_targets[b, 2:4, pos_mask_gt].T.cpu().numpy()
                    gt_cls = cls_targets[b, :, pos_mask_gt].T.argmax(dim=1).cpu().numpy()

                    # Convert cx, cy, bw, bh -> x1, y1, x2, y2 (in pixels 512x288)
                    for (cx, cy), (bw, bh), cid in zip(gt_cy_cx, gt_wh, gt_cls):
                        x1 = (cx - bw / 2.0) * 512.0
                        y1 = (cy - bh / 2.0) * 288.0
                        x2 = (cx + bw / 2.0) * 512.0
                        y2 = (cy + bh / 2.0) * 288.0
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_labels.append(cid)

                all_ground_truths.append({
                    'image_id': img_id,
                    'boxes': np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
                    'labels': np.array(gt_labels, dtype=np.int64) if gt_labels else np.empty((0,), dtype=np.int64)
                })

                # Extract Predictions for image above floor threshold
                cand_y, cand_x = torch.where(obj_prob[b, 0] >= eval_conf_floor)
                pred_boxes = []
                pred_scores = []
                pred_labels = []

                if len(cand_y) > 0:
                    cand_obj = obj_prob[b, 0, cand_y, cand_x]
                    cand_cls_scores, cand_cls_ids = cls_prob[b, :, cand_y, cand_x].max(dim=0)
                    final_scores = cand_obj * cand_cls_scores

                    cand_boxes = box_preds[b, :, cand_y, cand_x].T
                    cx = cand_boxes[:, 0] * 512.0
                    cy = cand_boxes[:, 1] * 288.0
                    bw = cand_boxes[:, 2] * 512.0
                    bh = cand_boxes[:, 3] * 288.0

                    x1 = cx - bw / 2.0
                    y1 = cy - bh / 2.0
                    x2 = cx + bw / 2.0
                    y2 = cy + bh / 2.0

                    cand_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

                    # Class-wise NMS
                    for c in range(len(CATEGORIES)):
                        c_mask = (cand_cls_ids == c)
                        if not c_mask.any():
                            continue
                        c_boxes = cand_boxes_xyxy[c_mask]
                        c_scores = final_scores[c_mask]
                        c_boxes_np = c_boxes.cpu().numpy()
                        c_scores_np = c_scores.cpu().numpy()
                        keep = nms(c_boxes_np, c_scores_np, iou_threshold=0.50)
                        for k in keep[:100]:
                            pred_boxes.append(c_boxes_np[k])
                            pred_scores.append(float(c_scores_np[k]))
                            pred_labels.append(c)

                all_predictions.append({
                    'image_id': img_id,
                    'boxes': np.array(pred_boxes, dtype=np.float32) if pred_boxes else np.empty((0, 4), dtype=np.float32),
                    'scores': np.array(pred_scores, dtype=np.float32) if pred_scores else np.empty((0,), dtype=np.float32),
                    'labels': np.array(pred_labels, dtype=np.int64) if pred_labels else np.empty((0,), dtype=np.int64)
                })

    evaluator = COCOEvaluator(class_names=CATEGORIES)
    eval_results = evaluator.evaluate(all_predictions, all_ground_truths, operational_conf_thresh=0.35)

    print("\n--- Corrected COCO Evaluation of Original Baseline Detector ---")
    print(json.dumps(eval_results, indent=2))

    out_path = 'reports/baseline_detector_corrected_eval.json'
    with open(out_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved report to: {out_path}")

if __name__ == '__main__':
    evaluate_baseline()
