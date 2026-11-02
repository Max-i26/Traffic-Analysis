import os
import sys
import xml.etree.ElementTree as ET
import cv2
import torch
import numpy as np
import motmetrics as mm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.detection.custom_fcos_detector import CustomFCOSDetector
from src.models.detection.fcos_decoder import FCOSDecoder
from src.data.fcos_loader import CLASS_NAMES
from src.tracking.tracker import ByteTracker
from src.tracking.kalman_filter import KalmanBoxTracker
from src.evaluation.coco_detection_evaluator import box_iou_matrix
from src.evaluation.evaluate_tracking_benchmark import load_detrac_xml, compute_hota

def test_calibrated_tracking(tau_high=0.12, tau_low=0.05, num_frames=50):
    xml_path = 'datasets/detrac_benchmark/annotations/MVI_39501.xml'
    img_dir = 'datasets/detrac_benchmark/MVI_39501'
    frames_gt, ignored_boxes = load_detrac_xml(xml_path, max_frames=num_frames)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcos_ckpt = 'checkpoints/fcos_v2_2_expanded.pt'
    fcos_model = CustomFCOSDetector(
        num_classes=len(CLASS_NAMES),
        fpn_channels=96,
        tower_channels=96,
        normalize_by_stride=True
    ).to(device)

    state = torch.load(fcos_ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    fcos_model.load_state_dict(state)
    fcos_model.eval()

    fcos_decoder = FCOSDecoder(
        strides=[4, 8, 16],
        score_thresh=0.01,
        pre_nms_top_k=500,
        post_nms_top_k=100,
        nms_iou_thresh=0.50,
        normalize_by_stride=True
    )

    KalmanBoxTracker.count = 0
    tracker = ByteTracker(
        tau_high=tau_high,
        tau_low=tau_low,
        match_thresh_stage1=0.40,
        match_thresh_stage2=0.30,
        max_age=15,
        min_hits=2
    )

    acc = mm.MOTAccumulator(auto_id=True)
    matches_by_frame = []
    gt_by_frame = []
    pr_by_frame = []

    scale_x = 960.0 / 512.0
    scale_y = 540.0 / 288.0

    with torch.no_grad():
        for f_idx in range(1, num_frames + 1):
            fname = f"img{f_idx:05d}.jpg"
            img_path = os.path.join(img_dir, fname)
            img_raw = cv2.imread(img_path)
            if img_raw is None:
                continue

            img_resized = cv2.resize(img_raw, (512, 288))
            img_tensor = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)

            preds = fcos_model(img_tensor)
            decoded = fcos_decoder.decode_predictions(preds, score_thresh=0.01, pre_nms_top_k=500)[0]

            det_boxes = decoded['boxes'].copy()
            det_scores = decoded['scores']
            det_labels = decoded['labels']

            if len(det_boxes) > 0:
                det_boxes[:, [0, 2]] *= scale_x
                det_boxes[:, [1, 3]] *= scale_y

                if len(ignored_boxes) > 0:
                    ign_ious = box_iou_matrix(det_boxes, ignored_boxes)
                    max_ign_overlap = ign_ious.max(axis=1)
                    keep_mask = max_ign_overlap < 0.50
                    det_boxes = det_boxes[keep_mask]
                    det_scores = det_scores[keep_mask]
                    det_labels = det_labels[keep_mask]

            tracks = tracker.update(det_boxes, det_scores, det_labels)

            gts = frames_gt.get(f_idx, [])
            gt_ids = [g['id'] for g in gts]
            gt_boxes = np.array([g['bbox'] for g in gts]) if gts else np.empty((0, 4))

            hyp_ids = [t['track_id'] for t in tracks]
            hyp_boxes = np.array([t['bbox'] for t in tracks]) if tracks else np.empty((0, 4))

            gt_by_frame.append(gt_ids)
            pr_by_frame.append(hyp_ids)

            if len(gt_boxes) > 0 and len(hyp_boxes) > 0:
                ious = box_iou_matrix(gt_boxes, hyp_boxes)
                dists = 1.0 - ious
                dists[ious < 0.40] = np.nan
                acc.update(gt_ids, hyp_ids, dists)
            elif len(gt_boxes) > 0:
                acc.update(gt_ids, [], np.empty((len(gt_ids), 0)))
            elif len(hyp_boxes) > 0:
                acc.update([], hyp_ids, np.empty((0, len(hyp_ids))))
            else:
                acc.update([], [], np.empty((0, 0)))

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['mota', 'motp', 'idf1', 'idp', 'idr', 'num_switches', 'num_fragmentations',
                 'num_false_positives', 'num_misses', 'num_objects', 'mostly_tracked', 'mostly_lost'],
        name=f'ByteTrack tau_high={tau_high}'
    )
    print("\nCalibrated Threshold Results:")
    print(summary)

if __name__ == '__main__':
    test_calibrated_tracking(tau_high=0.12, tau_low=0.05, num_frames=50)
