import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.fcos_loader import get_fcos_dataloaders
from src.models.detection.fcos_assigner import FCOSAssigner

def audit_regression_target_ranges():
    split_file = 'datasets/splits/bdd_expanded_splits.json'
    root_dir = 'datasets/extracted/bdd_pilot'
    train_loader, _ = get_fcos_dataloaders(split_file=split_file, root_dir=root_dir, batch_size=4, num_workers=0)

    assigner = FCOSAssigner(
        strides=[4, 8, 16],
        scale_ranges=[(0.0, 64.0), (64.0, 128.0), (128.0, 1e7)],
        center_radius=1.5,
        normalize_by_stride=True
    )

    feat_shapes = [(72, 128), (36, 64), (18, 32)]
    device = torch.device('cpu')
    points_per_level = assigner.generate_points(feat_shapes, device=device)

    all_dists = []
    level_dists = {4: [], 8: [], 16: []}

    num_pts_p2 = feat_shapes[0][0] * feat_shapes[0][1] # 9216
    num_pts_p3 = feat_shapes[1][0] * feat_shapes[1][1] # 2304
    num_pts_p4 = feat_shapes[2][0] * feat_shapes[2][1] # 576

    for imgs, targets in train_loader:
        for b in range(len(targets)):
            gt_b = targets[b]['boxes']
            gt_l = targets[b]['labels']
            cls_t, reg_t, cnt_t, pos_m = assigner.assign_targets_single_image(points_per_level, gt_b, gt_l)

            if pos_m.sum() == 0:
                continue

            pos_reg = reg_t[pos_m].numpy() # [K, 4]
            all_dists.extend(pos_reg.flatten())

            # Split by level
            pos_p2 = pos_m[:num_pts_p2]
            pos_p3 = pos_m[num_pts_p2:num_pts_p2 + num_pts_p3]
            pos_p4 = pos_m[num_pts_p2 + num_pts_p3:]

            if pos_p2.sum() > 0:
                level_dists[4].extend(reg_t[:num_pts_p2][pos_p2].numpy().flatten())
            if pos_p3.sum() > 0:
                level_dists[8].extend(reg_t[num_pts_p2:num_pts_p2 + num_pts_p3][pos_p3].numpy().flatten())
            if pos_p4.sum() > 0:
                level_dists[16].extend(reg_t[num_pts_p2 + num_pts_p3:][pos_p4].numpy().flatten())

    all_dists = np.array(all_dists)
    print("=" * 80)
    print("      REGRESSION TARGET RANGE AUDIT (FOR DFL REG_MAX=16)")
    print("=" * 80)
    print(f"Total positive target distances evaluated: {len(all_dists):,}")
    print(f"Overall Min: {all_dists.min():.2f} | Median: {np.median(all_dists):.2f} | 90th percentile: {np.percentile(all_dists, 90):.2f} | 95th: {np.percentile(all_dists, 95):.2f} | Max: {all_dists.max():.2f}")
    pct_within_16 = np.mean(all_dists <= 16.0) * 100.0
    print(f"Percentage of targets <= reg_max (16.0): {pct_within_16:.2f}%")

    for s in [4, 8, 16]:
        d = np.array(level_dists[s])
        if len(d) > 0:
            print(f"  Level Stride {s:>2} (N={len(d):,}): Median={np.median(d):.2f}, 95th%={np.percentile(d, 95):.2f}, Max={d.max():.2f}, <= 16: {np.mean(d <= 16.0)*100.0:.2f}%")

if __name__ == '__main__':
    audit_regression_target_ranges()
