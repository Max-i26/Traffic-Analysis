"""
Identity-Annotated Multi-Object Tracking Benchmark
Evaluates ByteTrack + 8-State Kalman Motion Model on the standard UA-DETRAC benchmark (MVI_39501).

Implements the user-directed two-tier scientific evaluation:
1. ORACLE EXPERIMENT: Ground-truth bounding boxes fed directly into ByteTrack.
   -> Isolates and evaluates the Kalman motion model + bipartite Hungarian association algorithm.
2. END-TO-END EXPERIMENT: Custom FCOS v2.2 real-time detections fed into ByteTrack.
   -> Evaluates complete real-world perception-to-tracking pipeline.

Metrics computed:
- MOTA (Multiple Object Tracking Accuracy)
- IDF1 (Identification F1 Score)
- HOTA (Higher Order Tracking Accuracy at alpha=0.50: sqrt(DetA * AssA))
- DetA (Detection Accuracy) & AssA (Association Accuracy)
- ID Switches (IDSW) & Track Fragmentations (Frag)
- Mostly Tracked (MT >= 80%) & Mostly Lost (ML <= 20%)
- False Positives (FP) & False Negatives / Misses (FN)
"""

import os
import sys
import xml.etree.ElementTree as ET
import cv2
import torch
import numpy as np
import json
import motmetrics as mm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.detection.custom_fcos_detector import CustomFCOSDetector
from src.models.detection.fcos_decoder import FCOSDecoder
from src.data.fcos_loader import CLASS_NAMES
from src.tracking.tracker import ByteTracker
from src.tracking.kalman_filter import KalmanBoxTracker
from src.evaluation.coco_detection_evaluator import box_iou_matrix

def load_detrac_xml(xml_path: str, max_frames: int = 100):
    """
    Parses UA-DETRAC sequence XML annotations and ignored regions.
    Returns:
        frames_gt: dict mapping frame_num (1..N) to list of {'id': int, 'bbox': [x1, y1, x2, y2], 'type': str}
        ignored_boxes: np.ndarray [M, 4] of non-evaluated road zones.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ignored_boxes = []
    ign_elem = root.find('ignored_region')
    if ign_elem is not None:
        for b in ign_elem.findall('box'):
            left = float(b.get('left'))
            top = float(b.get('top'))
            w = float(b.get('width'))
            h = float(b.get('height'))
            ignored_boxes.append([left, top, left + w, top + h])
    ignored_boxes = np.array(ignored_boxes) if ignored_boxes else np.empty((0, 4))

    frames_gt = {}
    for f in root.findall('frame'):
        num = int(f.get('num'))
        if num > max_frames:
            continue
        gts = []
        tlist = f.find('target_list')
        if tlist is not None:
            for t in tlist.findall('target'):
                tid = int(t.get('id'))
                b = t.find('box')
                left = float(b.get('left'))
                top = float(b.get('top'))
                w = float(b.get('width'))
                h = float(b.get('height'))
                attr = t.find('attribute')
                vtype = attr.get('vehicle_type') if attr is not None else 'car'
                gts.append({
                    'id': tid,
                    'bbox': np.array([left, top, left + w, top + h], dtype=np.float32),
                    'type': vtype
                })
        frames_gt[num] = gts

    return frames_gt, ignored_boxes

def compute_hota(matches_by_frame, all_gt_by_frame, all_pr_by_frame):
    """
    Computes Higher Order Tracking Accuracy (HOTA) at alpha=0.50
    following Luiten et al. (IJCV 2020).
    matches_by_frame: list of list of (gt_id, pr_id)
    """
    # Total TPs, FNs, FPs
    total_tp = sum(len(m) for m in matches_by_frame)
    total_gt = sum(len(g) for g in all_gt_by_frame)
    total_pr = sum(len(p) for p in all_pr_by_frame)

    total_fn = total_gt - total_tp
    total_fp = total_pr - total_tp

    deta = total_tp / max(1e-7, total_tp + total_fn + total_fp)

    if total_tp == 0:
        return {'HOTA': 0.0, 'DetA': 0.0, 'AssA': 0.0}

    # Association alignment: for each (gt, pr) match, count temporal overlap
    # Map (gt_id, pr_id) -> count of co-occurrences
    pair_counts = {}
    gt_match_counts = {}
    pr_match_counts = {}

    for frame_matches in matches_by_frame:
        for gid, pid in frame_matches:
            pair_counts[(gid, pid)] = pair_counts.get((gid, pid), 0) + 1
            gt_match_counts[gid] = gt_match_counts.get(gid, 0) + 1
            pr_match_counts[pid] = pr_match_counts.get(pid, 0) + 1

    # For each TP match c in the dataset:
    # TPA = pair_counts[(gid, pid)]
    # FNA = total appearances of gid not with pid = gt_total_appearances - TPA
    # FPA = total appearances of pid not with gid = pr_total_appearances - TPA
    # AssA(c) = TPA / (TPA + FNA + FPA) = TPA / (gt_app + pr_app - TPA)
    gt_total_appearances = {}
    for frame_gts in all_gt_by_frame:
        for gid in frame_gts:
            gt_total_appearances[gid] = gt_total_appearances.get(gid, 0) + 1

    pr_total_appearances = {}
    for frame_prs in all_pr_by_frame:
        for pid in frame_prs:
            pr_total_appearances[pid] = pr_total_appearances.get(pid, 0) + 1

    assa_scores = []
    for frame_matches in matches_by_frame:
        for gid, pid in frame_matches:
            tpa = pair_counts[(gid, pid)]
            fna = gt_total_appearances.get(gid, tpa) - tpa
            fpa = pr_total_appearances.get(pid, tpa) - tpa
            denom = tpa + fna + fpa
            assa_scores.append(tpa / max(1e-7, denom))

    assa = np.mean(assa_scores) if assa_scores else 0.0
    hota = np.sqrt(deta * assa)

    return {
        'HOTA': round(float(hota), 4),
        'DetA': round(float(deta), 4),
        'AssA': round(float(assa), 4)
    }

def run_tracking_benchmark(num_frames: int = 100):
    xml_path = 'datasets/detrac_benchmark/annotations/MVI_39501.xml'
    img_dir = 'datasets/detrac_benchmark/MVI_39501'

    frames_gt, ignored_boxes = load_detrac_xml(xml_path, max_frames=num_frames)
    print(f"Loaded {len(frames_gt)} frames from UA-DETRAC sequence MVI_39501.")
    print(f"Ignored boundary regions: {len(ignored_boxes)} zones.")

    # =========================================================================
    # EXPERIMENT 1: ORACLE DETECTIONS EXPERIMENT
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: ORACLE TRACKER EVALUATION (PERFECT DETECTOR INPUT)")
    print("Target: Isolates and benchmarks 8-state Kalman Filter + ByteTrack association")
    print("=" * 80)

    KalmanBoxTracker.count = 0
    oracle_tracker = ByteTracker(
        tau_high=0.35,
        tau_low=0.10,
        match_thresh_stage1=0.50,
        match_thresh_stage2=0.40,
        max_age=15,
        min_hits=1  # Immediate track confirmation for oracle evaluation
    )

    acc_oracle = mm.MOTAccumulator(auto_id=True)
    oracle_matches_by_frame = []
    oracle_gt_by_frame = []
    oracle_pr_by_frame = []

    for f_idx in range(1, num_frames + 1):
        gts = frames_gt.get(f_idx, [])
        gt_ids = [g['id'] for g in gts]
        gt_boxes = np.array([g['bbox'] for g in gts]) if gts else np.empty((0, 4))

        # In Oracle mode, ground truth boxes are fed as detections with score 1.0
        if len(gt_boxes) > 0:
            det_scores = np.ones(len(gt_boxes), dtype=np.float32)
            det_labels = np.zeros(len(gt_boxes), dtype=int)
            tracks = oracle_tracker.update(gt_boxes, det_scores, det_labels)
        else:
            tracks = oracle_tracker.update(np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int))

        hyp_ids = [t['track_id'] for t in tracks]
        hyp_boxes = np.array([t['bbox'] for t in tracks]) if tracks else np.empty((0, 4))

        oracle_gt_by_frame.append(gt_ids)
        oracle_pr_by_frame.append(hyp_ids)

        if len(gt_boxes) > 0 and len(hyp_boxes) > 0:
            ious = box_iou_matrix(gt_boxes, hyp_boxes)
            dists = 1.0 - ious
            # Hungarian matching at IoU 0.50 threshold
            dists_match = dists.copy()
            dists_match[ious < 0.50] = 1000.0
            from scipy.optimize import linear_sum_assignment
            r_idx, c_idx = linear_sum_assignment(dists_match)
            frame_m = []
            for r, c in zip(r_idx, c_idx):
                if ious[r, c] >= 0.50:
                    frame_m.append((gt_ids[r], hyp_ids[c]))
            oracle_matches_by_frame.append(frame_m)

            # Mask out dists > 0.50 for motmetrics
            dists[ious < 0.50] = np.nan
            acc_oracle.update(gt_ids, hyp_ids, dists)
        elif len(gt_boxes) > 0:
            oracle_matches_by_frame.append([])
            acc_oracle.update(gt_ids, [], np.empty((len(gt_ids), 0)))
        elif len(hyp_boxes) > 0:
            oracle_matches_by_frame.append([])
            acc_oracle.update([], hyp_ids, np.empty((0, len(hyp_ids))))
        else:
            oracle_matches_by_frame.append([])
            acc_oracle.update([], [], np.empty((0, 0)))

    mh = mm.metrics.create()
    summary_oracle = mh.compute(
        acc_oracle,
        metrics=['mota', 'motp', 'idf1', 'idp', 'idr', 'num_switches', 'num_fragmentations',
                 'num_false_positives', 'num_misses', 'num_objects', 'mostly_tracked', 'mostly_lost'],
        name='Oracle ByteTrack'
    )
    hota_oracle = compute_hota(oracle_matches_by_frame, oracle_gt_by_frame, oracle_pr_by_frame)

    print("\n--- Oracle Experiment Results ---")
    print(f"  MOTA:       {summary_oracle['mota'].iloc[0]*100:.2f}%")
    print(f"  IDF1:       {summary_oracle['idf1'].iloc[0]*100:.2f}% (Precision: {summary_oracle['idp'].iloc[0]*100:.2f}%, Recall: {summary_oracle['idr'].iloc[0]*100:.2f}%)")
    print(f"  HOTA:       {hota_oracle['HOTA']*100:.2f}% (DetA: {hota_oracle['DetA']*100:.2f}%, AssA: {hota_oracle['AssA']*100:.2f}%)")
    print(f"  ID Switches: {int(summary_oracle['num_switches'].iloc[0])}")
    print(f"  Frag:       {int(summary_oracle['num_fragmentations'].iloc[0])}")
    print(f"  MT / ML:    {int(summary_oracle['mostly_tracked'].iloc[0])} mostly tracked / {int(summary_oracle['mostly_lost'].iloc[0])} mostly lost (Total GT: {int(summary_oracle['num_objects'].iloc[0])})")

    # =========================================================================
    # EXPERIMENT 2: END-TO-END SYSTEM (CUSTOM FCOS v2.2 DETECTIONS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: END-TO-END TRACKING EVALUATION (FCOS v2.2 + BYTETRACK)")
    print("Target: Evaluates actual deployed autonomous perception system")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcos_ckpt = 'checkpoints/fcos_v2_2_expanded.pt'
    fcos_model = CustomFCOSDetector(
        num_classes=len(CLASS_NAMES),
        fpn_channels=96,
        tower_channels=96,
        normalize_by_stride=True
    ).to(device)

    # Safe loading
    state = torch.load(fcos_ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    fcos_model.load_state_dict(state)
    fcos_model.eval()

    fcos_decoder = FCOSDecoder(
        strides=[4, 8, 16],
        score_thresh=0.05,
        pre_nms_top_k=500,
        post_nms_top_k=100,
        nms_iou_thresh=0.50,
        normalize_by_stride=True
    )

    KalmanBoxTracker.count = 0
    e2e_tracker = ByteTracker(
        tau_high=0.35,
        tau_low=0.10,
        match_thresh_stage1=0.50,
        match_thresh_stage2=0.40,
        max_age=15,
        min_hits=2
    )

    acc_e2e = mm.MOTAccumulator(auto_id=True)
    e2e_matches_by_frame = []
    e2e_gt_by_frame = []
    e2e_pr_by_frame = []

    target_w, target_h = 512, 288
    orig_w, orig_h = 960, 540

    scale_x = orig_w / target_w
    scale_y = orig_h / target_h

    with torch.no_grad():
        for f_idx in range(1, num_frames + 1):
            fname = f"img{f_idx:05d}.jpg"
            img_path = os.path.join(img_dir, fname)
            img_raw = cv2.imread(img_path)
            if img_raw is None:
                continue

            # Preprocessing
            img_resized = cv2.resize(img_raw, (target_w, target_h))
            img_tensor = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)

            # Inference & GPU Decode
            preds = fcos_model(img_tensor)
            decoded = fcos_decoder.decode_predictions(preds, score_thresh=0.05, pre_nms_top_k=500)[0]

            det_boxes_fcos = decoded['boxes']
            det_scores_fcos = decoded['scores']
            det_labels_fcos = decoded['labels']

            # Rescale boxes back to native 960x540
            if len(det_boxes_fcos) > 0:
                det_boxes_native = det_boxes_fcos.copy()
                det_boxes_native[:, [0, 2]] *= scale_x
                det_boxes_native[:, [1, 3]] *= scale_y

                # Filter out detections located heavily inside ignored regions
                if len(ignored_boxes) > 0:
                    ign_ious = box_iou_matrix(det_boxes_native, ignored_boxes)
                    max_ign_overlap = ign_ious.max(axis=1)
                    keep_mask = max_ign_overlap < 0.60
                    det_boxes_native = det_boxes_native[keep_mask]
                    det_scores_fcos = det_scores_fcos[keep_mask]
                    det_labels_fcos = det_labels_fcos[keep_mask]
            else:
                det_boxes_native = np.empty((0, 4))

            # Update ByteTracker
            tracks = e2e_tracker.update(det_boxes_native, det_scores_fcos, det_labels_fcos)

            # Ground truth for this frame
            gts = frames_gt.get(f_idx, [])
            gt_ids = [g['id'] for g in gts]
            gt_boxes = np.array([g['bbox'] for g in gts]) if gts else np.empty((0, 4))

            hyp_ids = [t['track_id'] for t in tracks]
            hyp_boxes = np.array([t['bbox'] for t in tracks]) if tracks else np.empty((0, 4))

            e2e_gt_by_frame.append(gt_ids)
            e2e_pr_by_frame.append(hyp_ids)

            if len(gt_boxes) > 0 and len(hyp_boxes) > 0:
                ious = box_iou_matrix(gt_boxes, hyp_boxes)
                dists = 1.0 - ious

                dists_match = dists.copy()
                dists_match[ious < 0.50] = 1000.0
                from scipy.optimize import linear_sum_assignment
                r_idx, c_idx = linear_sum_assignment(dists_match)
                frame_m = []
                for r, c in zip(r_idx, c_idx):
                    if ious[r, c] >= 0.50:
                        frame_m.append((gt_ids[r], hyp_ids[c]))
                e2e_matches_by_frame.append(frame_m)

                dists[ious < 0.50] = np.nan
                acc_e2e.update(gt_ids, hyp_ids, dists)
            elif len(gt_boxes) > 0:
                e2e_matches_by_frame.append([])
                acc_e2e.update(gt_ids, [], np.empty((len(gt_ids), 0)))
            elif len(hyp_boxes) > 0:
                e2e_matches_by_frame.append([])
                acc_e2e.update([], hyp_ids, np.empty((0, len(hyp_ids))))
            else:
                e2e_matches_by_frame.append([])
                acc_e2e.update([], [], np.empty((0, 0)))

    summary_e2e = mh.compute(
        acc_e2e,
        metrics=['mota', 'motp', 'idf1', 'idp', 'idr', 'num_switches', 'num_fragmentations',
                 'num_false_positives', 'num_misses', 'num_objects', 'mostly_tracked', 'mostly_lost'],
        name='End-to-End FCOS+ByteTrack'
    )
    hota_e2e = compute_hota(e2e_matches_by_frame, e2e_gt_by_frame, e2e_pr_by_frame)

    print("\n--- End-to-End System Results ---")
    print(f"  MOTA:       {summary_e2e['mota'].iloc[0]*100:.2f}%")
    print(f"  IDF1:       {summary_e2e['idf1'].iloc[0]*100:.2f}% (Precision: {summary_e2e['idp'].iloc[0]*100:.2f}%, Recall: {summary_e2e['idr'].iloc[0]*100:.2f}%)")
    print(f"  HOTA:       {hota_e2e['HOTA']*100:.2f}% (DetA: {hota_e2e['DetA']*100:.2f}%, AssA: {hota_e2e['AssA']*100:.2f}%)")
    print(f"  ID Switches: {int(summary_e2e['num_switches'].iloc[0])}")
    print(f"  Frag:       {int(summary_e2e['num_fragmentations'].iloc[0])}")
    print(f"  MT / ML:    {int(summary_e2e['mostly_tracked'].iloc[0])} mostly tracked / {int(summary_e2e['mostly_lost'].iloc[0])} mostly lost (Total GT: {int(summary_e2e['num_objects'].iloc[0])})")

    # =========================================================================
    # SUMMARY COMPARISON TABLE & EXPORT
    # =========================================================================
    results = {
        'benchmark_dataset': 'UA-DETRAC MVI_39501 (100 continuous traffic frames)',
        'oracle_tracker': {
            'MOTA': round(float(summary_oracle['mota'].iloc[0]), 4),
            'MOTP_mean_iou': round(1.0 - float(summary_oracle['motp'].iloc[0]), 4),
            'IDF1': round(float(summary_oracle['idf1'].iloc[0]), 4),
            'IDP': round(float(summary_oracle['idp'].iloc[0]), 4),
            'IDR': round(float(summary_oracle['idr'].iloc[0]), 4),
            'HOTA': hota_oracle['HOTA'],
            'DetA': hota_oracle['DetA'],
            'AssA': hota_oracle['AssA'],
            'id_switches': int(summary_oracle['num_switches'].iloc[0]),
            'fragmentations': int(summary_oracle['num_fragmentations'].iloc[0]),
            'false_positives': int(summary_oracle['num_false_positives'].iloc[0]),
            'misses': int(summary_oracle['num_misses'].iloc[0]),
            'mostly_tracked': int(summary_oracle['mostly_tracked'].iloc[0]),
            'mostly_lost': int(summary_oracle['mostly_lost'].iloc[0]),
            'total_ground_truth_instances': int(summary_oracle['num_objects'].iloc[0])
        },
        'end_to_end_pipeline': {
            'MOTA': round(float(summary_e2e['mota'].iloc[0]), 4),
            'MOTP_mean_iou': round(1.0 - float(summary_e2e['motp'].iloc[0]), 4),
            'IDF1': round(float(summary_e2e['idf1'].iloc[0]), 4),
            'IDP': round(float(summary_e2e['idp'].iloc[0]), 4),
            'IDR': round(float(summary_e2e['idr'].iloc[0]), 4),
            'HOTA': hota_e2e['HOTA'],
            'DetA': hota_e2e['DetA'],
            'AssA': hota_e2e['AssA'],
            'id_switches': int(summary_e2e['num_switches'].iloc[0]),
            'fragmentations': int(summary_e2e['num_fragmentations'].iloc[0]),
            'false_positives': int(summary_e2e['num_false_positives'].iloc[0]),
            'misses': int(summary_e2e['num_misses'].iloc[0]),
            'mostly_tracked': int(summary_e2e['mostly_tracked'].iloc[0]),
            'mostly_lost': int(summary_e2e['mostly_lost'].iloc[0]),
            'total_ground_truth_instances': int(summary_e2e['num_objects'].iloc[0])
        }
    }

    report_path = 'reports/tracking_benchmark_mota_idf1.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Identity tracking benchmark saved to: {report_path}")
    return results

if __name__ == '__main__':
    run_tracking_benchmark(num_frames=100)
