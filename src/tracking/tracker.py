"""
ByteTrack Multi-Object Vehicle Tracker.
Implements two-stage IoU association:
1. Stage 1: Associate active tracks with high-confidence detections (score >= tau_high).
2. Stage 2: Associate remaining tracks with low-confidence detections (tau_low <= score < tau_high).
Prevents track fragmentation during occlusions while avoiding false-positive track initiation.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment

from src.tracking.kalman_filter import KalmanBoxTracker
from src.evaluation.coco_detection_evaluator import box_iou_matrix

class ByteTracker:
    def __init__(self,
                 tau_high: float = 0.35,
                 tau_low: float = 0.10,
                 match_thresh_stage1: float = 0.50,
                 match_thresh_stage2: float = 0.40,
                 max_age: int = 15,
                 min_hits: int = 2):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.match_thresh_stage1 = match_thresh_stage1
        self.match_thresh_stage2 = match_thresh_stage2
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracked_tracks: List[KalmanBoxTracker] = []
        self.lost_tracks: List[KalmanBoxTracker] = []
        self.frame_id = 0

    def update(self,
               det_boxes: np.ndarray,
               det_scores: np.ndarray,
               det_labels: np.ndarray) -> List[Dict]:
        """
        det_boxes: [N, 4] (x1, y1, x2, y2)
        det_scores: [N] confidence scores
        det_labels: [N] class IDs
        Returns list of active confirmed track dicts:
            {
               'track_id': int,
               'bbox': [x1, y1, x2, y2],
               'score': float,
               'class_id': int,
               'velocity': [vx, vy],
               'hits': int,
               'age': int
            }
        """
        self.frame_id += 1

        # 1. Predict new locations of existing tracks
        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()

        # 2. Separate detections into High and Low confidence
        if len(det_boxes) > 0:
            mask_high = det_scores >= self.tau_high
            mask_low = (det_scores >= self.tau_low) & (det_scores < self.tau_high)

            boxes_high = det_boxes[mask_high]
            scores_high = det_scores[mask_high]
            labels_high = det_labels[mask_high]

            boxes_low = det_boxes[mask_low]
            scores_low = det_scores[mask_low]
            labels_low = det_labels[mask_low]
        else:
            boxes_high = np.empty((0, 4))
            scores_high = np.empty((0,))
            labels_high = np.empty((0,), dtype=int)
            boxes_low = np.empty((0, 4))
            scores_low = np.empty((0,))
            labels_low = np.empty((0,), dtype=int)

        # Pool active and recently lost tracks for Stage 1 association
        track_pool = self.tracked_tracks + self.lost_tracks

        # =====================================================================
        # STAGE 1: Match Track Pool with High-Confidence Detections
        # =====================================================================
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_high = self._match(
            track_pool, boxes_high, match_thresh=self.match_thresh_stage1
        )

        for t_idx, d_idx in matched_tracks_1:
            track = track_pool[t_idx]
            track.update(boxes_high[d_idx], scores_high[d_idx], labels_high[d_idx])

        # Remaining tracks to consider for Stage 2
        remain_tracks = [track_pool[i] for i in unmatched_tracks_1 if track_pool[i].state == 'tracked']

        # =====================================================================
        # STAGE 2: Match Remaining Tracks with Low-Confidence Detections
        # =====================================================================
        matched_tracks_2, unmatched_tracks_2, _ = self._match(
            remain_tracks, boxes_low, match_thresh=self.match_thresh_stage2
        )

        for t_idx, d_idx in matched_tracks_2:
            track = remain_tracks[t_idx]
            track.update(boxes_low[d_idx], scores_low[d_idx], labels_low[d_idx])

        # Tracks that still failed to match are marked as lost
        final_unmatched_tracks = [remain_tracks[i] for i in unmatched_tracks_2]
        for t in final_unmatched_tracks:
            t.state = 'lost'

        # =====================================================================
        # STAGE 3: Initialize New Tracks for Unmatched High-Confidence Detections
        # =====================================================================
        new_tracks = []
        for d_idx in unmatched_dets_high:
            # Only spawn track if detection score is strong
            if scores_high[d_idx] >= self.tau_high:
                new_track = KalmanBoxTracker(
                    bbox=boxes_high[d_idx],
                    score=scores_high[d_idx],
                    class_id=labels_high[d_idx]
                )
                new_tracks.append(new_track)

        # Update track collections
        self.tracked_tracks = [t for t in track_pool if t.state == 'tracked' and t.time_since_update == 0] + new_tracks
        self.lost_tracks = [t for t in track_pool if t.state == 'lost' and t.time_since_update <= self.max_age]

        # Gather active confirmed tracks for output
        active_outputs = []
        for t in self.tracked_tracks:
            # Require min_hits to suppress transient single-frame false alarms
            if t.hits >= self.min_hits or self.frame_id <= self.min_hits:
                state_box = t.get_state()
                vel = t.get_velocity()
                active_outputs.append({
                    'track_id': t.id,
                    'bbox': state_box,
                    'score': float(t.score),
                    'class_id': int(t.class_id),
                    'velocity': vel,
                    'hits': t.hits,
                    'age': t.age
                })

        return active_outputs

    def _match(self,
               tracks: List[KalmanBoxTracker],
               detections: np.ndarray,
               match_thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Bipartite matching via Hungarian algorithm on IoU cost matrix.
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.get_state() for t in tracks])
        ious = box_iou_matrix(track_boxes, detections)
        cost_matrix = 1.0 - ious

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if ious[r, c] >= match_thresh:
                matched_pairs.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched_pairs, list(unmatched_tracks), list(unmatched_dets)
