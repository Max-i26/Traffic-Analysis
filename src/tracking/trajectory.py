"""
Vehicle Trajectory Buffer, Motion Velocity Estimation, and Lane Corridor Mapping.
Estimates pixel velocity vectors v_pixel = (dx/dt, dy/dt) and maps vehicle base centroids
into lane corridors (Host Lane, Left Lane, Right Lane).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class TrajectoryManager:
    def __init__(self, max_history: int = 30, fps: float = 20.0):
        self.max_history = max_history
        self.fps = fps
        # Map: track_id -> list of (x, y) centroids
        self.trajectories: Dict[int, List[Tuple[float, float]]] = {}
        # Default lateral lane boundaries in normalized coordinates [0, 1] across image width
        # Left Passing Lane: [0.15, 0.38], Host Lane: [0.38, 0.62], Right Shoulder/Passing: [0.62, 0.85]
        self.lane_corridors = {
            'L1 (Left Lane)': (0.15, 0.38),
            'Host Lane': (0.38, 0.62),
            'R1 (Right Lane)': (0.62, 0.88)
        }

    def update(self, tracks: List[Dict], img_width: int = 512, img_height: int = 288):
        """
        Updates trajectory buffers and computes lane assignment and pixel velocity.
        """
        current_ids = set()

        for t in tracks:
            tid = t['track_id']
            current_ids.add(tid)
            bbox = t['bbox']
            # Vehicle base center is the ground-contact contact point
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = bbox[3]  # Bottom edge of bounding box

            if tid not in self.trajectories:
                self.trajectories[tid] = []
            self.trajectories[tid].append((cx, cy))

            if len(self.trajectories[tid]) > self.max_history:
                self.trajectories[tid].pop(0)

            # 1. Pixel Velocity Vector v_pixel = (dx/dt, dy/dt)
            # dt = time across historical samples
            pts = self.trajectories[tid]
            if len(pts) >= 3:
                dx = (pts[-1][0] - pts[0][0])
                dy = (pts[-1][1] - pts[0][1])
                dt = (len(pts) - 1) / self.fps
                vx = dx / max(1e-4, dt)
                vy = dy / max(1e-4, dt)
            else:
                vx, vy = 0.0, 0.0

            t['v_pixel'] = (round(float(vx), 1), round(float(vy), 1))
            t['trajectory'] = list(self.trajectories[tid])

            # 2. Lane Corridor Assignment
            norm_cx = cx / float(img_width)
            assigned_lane = "Off-road / Shoulder"
            for lane_name, (x_min, x_max) in self.lane_corridors.items():
                if x_min <= norm_cx < x_max:
                    assigned_lane = lane_name
                    break
            t['lane'] = assigned_lane

        # Cleanup tracks that disappeared
        dead_ids = [tid for tid in self.trajectories if tid not in current_ids]
        for tid in dead_ids:
            if len(self.trajectories[tid]) > 0:
                self.trajectories[tid].pop(0)
            if len(self.trajectories[tid]) == 0:
                del self.trajectories[tid]

    def get_lane_occupancy_counts(self, tracks: List[Dict]) -> Dict[str, int]:
        """Returns vehicle count per lane corridor."""
        counts = {name: 0 for name in self.lane_corridors}
        counts['Off-road / Shoulder'] = 0
        for t in tracks:
            lane = t.get('lane', 'Off-road / Shoulder')
            counts[lane] = counts.get(lane, 0) + 1
        return counts
