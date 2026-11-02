"""
FCOS Target Assigner.
Implements:
1. Multi-scale scale range assignment:
   - P2 (stride 4): [0, 64] px
   - P3 (stride 8): [64, 128] px
   - P4 (stride 16): [128, inf) px
2. Center sampling: Positive points must lie within center radius r * stride.
3. Ambiguity resolution: Points claimed by multiple GT boxes are assigned to the box with minimal area.
4. Generates classification targets, LTRB regression targets, and centerness targets.
"""

import math
import torch
import numpy as np
from typing import List, Dict, Tuple

class FCOSAssigner:
    def __init__(self,
                 strides: List[int] = [4, 8, 16],
                 scale_ranges: List[Tuple[float, float]] = [(0.0, 64.0), (64.0, 128.0), (128.0, 1e7)],
                 center_radius: float = 1.5,
                 normalize_by_stride: bool = True):
        self.strides = strides
        self.scale_ranges = scale_ranges
        self.center_radius = center_radius
        self.normalize_by_stride = normalize_by_stride

    def generate_points(self, feat_shapes: List[Tuple[int, int]], device: torch.device) -> List[torch.Tensor]:
        """
        Generates (x, y) center coordinates for each cell in each feature pyramid level.
        Returns: list of [H_i * W_i, 2] tensors.
        """
        all_points = []
        for stride, (H, W) in zip(self.strides, feat_shapes):
            shift_x = (torch.arange(0, W, device=device) + 0.5) * stride
            shift_y = (torch.arange(0, H, device=device) + 0.5) * stride
            y, x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            points = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1) # [H*W, 2]
            all_points.append(points)
        return all_points

    def assign_targets_single_image(self,
                                    points_per_level: List[torch.Tensor],
                                    gt_boxes: torch.Tensor,
                                    gt_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assigns targets for a single image across all feature pyramid levels.
        gt_boxes: [N, 4] in (x1, y1, x2, y2)
        gt_labels: [N] category indices (0 to num_classes - 1)

        Returns:
            cls_targets: [Total_Points] (-1 for background, >=0 for class id)
            reg_targets: [Total_Points, 4] (l, t, r, b)
            cnt_targets: [Total_Points] (centerness in [0, 1])
            pos_masks:   [Total_Points] (bool)
        """
        total_points = sum(p.size(0) for p in points_per_level)
        device = points_per_level[0].device

        if gt_boxes.size(0) == 0:
            cls_targets = torch.full((total_points,), -1, dtype=torch.long, device=device)
            reg_targets = torch.zeros((total_points, 4), dtype=torch.float32, device=device)
            cnt_targets = torch.zeros((total_points,), dtype=torch.float32, device=device)
            pos_masks = torch.zeros((total_points,), dtype=torch.bool, device=device)
            return cls_targets, reg_targets, cnt_targets, pos_masks

        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) # [N]

        level_cls_targets = []
        level_reg_targets = []
        level_cnt_targets = []
        level_pos_masks = []

        for level_idx, (points, stride, (min_size, max_size)) in enumerate(
            zip(points_per_level, self.strides, self.scale_ranges)
        ):
            M = points.size(0)
            N = gt_boxes.size(0)

            # points: [M, 2], gt_boxes: [N, 4]
            # Compute l, t, r, b for all points and all gt_boxes: [M, N]
            xs = points[:, 0].unsqueeze(1) # [M, 1]
            ys = points[:, 1].unsqueeze(1) # [M, 1]

            l = xs - gt_boxes[:, 0].unsqueeze(0) # [M, N]
            t = ys - gt_boxes[:, 1].unsqueeze(0) # [M, N]
            r = gt_boxes[:, 2].unsqueeze(0) - xs # [M, N]
            b = gt_boxes[:, 3].unsqueeze(0) - ys # [M, N]

            ltrb = torch.stack([l, t, r, b], dim=-1) # [M, N, 4]

            # 1. Point must be inside the bounding box
            is_in_box = ltrb.min(dim=-1)[0] > 0 # [M, N]

            # 2. Scale range check: max(l, t, r, b) in [min_size, max_size)
            max_dist = ltrb.max(dim=-1)[0] # [M, N]
            is_in_scale = (max_dist >= min_size) & (max_dist < max_size) # [M, N]

            # 3. Center sampling: point must be inside center sub-box
            cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
            cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
            c_radius = self.center_radius * stride

            c_x1 = torch.maximum(gt_boxes[:, 0], cx - c_radius)
            c_y1 = torch.maximum(gt_boxes[:, 1], cy - c_radius)
            c_x2 = torch.minimum(gt_boxes[:, 2], cx + c_radius)
            c_y2 = torch.minimum(gt_boxes[:, 3], cy + c_radius)

            c_l = xs - c_x1.unsqueeze(0)
            c_t = ys - c_y1.unsqueeze(0)
            c_r = c_x2.unsqueeze(0) - xs
            c_b = c_y2.unsqueeze(0) - ys
            is_in_center = torch.stack([c_l, c_t, c_r, c_b], dim=-1).min(dim=-1)[0] > 0 # [M, N]

            # Eligible positive mask: [M, N]
            is_pos = is_in_box & is_in_scale & is_in_center

            # Ambiguity resolution: If point satisfies multiple GTs, pick one with MINIMUM AREA
            areas_expanded = areas.unsqueeze(0).repeat(M, 1) # [M, N]
            areas_expanded[~is_pos] = 1e10 # Ineligible pairs get huge area

            min_area, best_gt_idx = areas_expanded.min(dim=1) # [M]
            point_is_pos = min_area < 1e9 # [M]

            # Gather targets for points
            assigned_gt = best_gt_idx # [M]
            assigned_ltrb = ltrb[torch.arange(M), assigned_gt] # [M, 4]
            assigned_cls = gt_labels[assigned_gt] # [M]
            assigned_cls[~point_is_pos] = -1 # Background

            # Centerness target: sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
            al = assigned_ltrb[:, 0]
            at = assigned_ltrb[:, 1]
            ar = assigned_ltrb[:, 2]
            ab = assigned_ltrb[:, 3]

            lr_min = torch.minimum(al, ar)
            lr_max = torch.maximum(al, ar)
            tb_min = torch.minimum(at, ab)
            tb_max = torch.maximum(at, ab)

            cnt = torch.sqrt((lr_min / torch.clamp(lr_max, min=1e-6)) * (tb_min / torch.clamp(tb_max, min=1e-6)))
            cnt[~point_is_pos] = 0.0

            # Experiment A: Stride-normalize regression targets to dimensionless values
            if self.normalize_by_stride:
                assigned_ltrb = assigned_ltrb / float(stride)

            level_cls_targets.append(assigned_cls)
            level_reg_targets.append(assigned_ltrb)
            level_cnt_targets.append(cnt)
            level_pos_masks.append(point_is_pos)

        return (
            torch.cat(level_cls_targets, dim=0),
            torch.cat(level_reg_targets, dim=0),
            torch.cat(level_cnt_targets, dim=0),
            torch.cat(level_pos_masks, dim=0)
        )
