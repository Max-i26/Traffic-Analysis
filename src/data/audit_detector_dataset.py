import os
import sys
import json
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

SPLIT_PATH = 'datasets/splits/bdd_pilot_splits.json'
BDD_ROOT_DIR = 'datasets/extracted/bdd_pilot'
REPORT_JSON_PATH = 'reports/detector_dataset_audit.json'
VISUAL_DIR = 'reports/visual_samples'

TARGET_W, TARGET_H = 512, 288
CATEGORIES = ['car', 'bus', 'truck', 'motorcycle']
CLASS_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}

def run_audit():
    print("=" * 70)
    print("STAGE 1: DATASET INTEGRITY & GRID COLLISION AUDIT")
    print("=" * 70)

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Split file not found: {SPLIT_PATH}")

    with open(SPLIT_PATH, 'r') as f:
        splits = json.load(f)

    os.makedirs(os.path.dirname(REPORT_JSON_PATH), exist_ok=True)
    os.makedirs(VISUAL_DIR, exist_ok=True)

    audit_summary = {}

    for split_name in ['train', 'val']:
        records = splits[split_name]
        print(f"\n--- Auditing Split: '{split_name}' ({len(records)} images) ---")

        total_images = len(records)
        existing_images = 0
        total_vehicles = 0
        degenerate_boxes = 0
        clipped_boxes = 0

        class_counts = {c: 0 for c in CATEGORIES}
        other_classes = {}

        # Scale counts (target resolution 512x288)
        scale_counts = {'small (<32^2)': 0, 'medium (32^2 - 96^2)': 0, 'large (>96^2)': 0}

        # Grid collision tracking for single-scale stride 8
        stride = 8
        grid_w, grid_h = TARGET_W // stride, TARGET_H // stride # 64 x 36
        images_with_collision_s8 = 0
        total_overwritten_s8 = 0

        # Grid collision tracking for multi-scale FPN (stride 4, 8, 16)
        # Scale ranges in 512x288:
        # P2 (stride 4, 128x72): max_dim < 48
        # P3 (stride 8,  64x36): 40 <= max_dim < 128
        # P4 (stride 16, 32x18): max_dim >= 96
        images_with_collision_fpn = 0
        total_overwritten_fpn = 0

        for idx, rec in enumerate(records):
            img_rel = rec['image_path'].replace('\\', '/')
            img_full = os.path.join(BDD_ROOT_DIR, *img_rel.split('/'))

            if not os.path.exists(img_full):
                continue
            existing_images += 1

            orig_w = rec.get('width', 1280)
            orig_h = rec.get('height', 720)
            scale_x = TARGET_W / orig_w
            scale_y = TARGET_H / orig_h

            vehicles = rec.get('vehicles', [])
            total_vehicles += len(vehicles)

            # Cell occupants for this image: (gy, gx) -> list of vehicle indices
            cell_occupants_s8 = {}

            # Multi-scale occupants: (level, gy, gx) -> list
            cell_occupants_fpn = {}

            for v_idx, veh in enumerate(vehicles):
                cat = veh.get('category', 'car')
                if cat in class_counts:
                    class_counts[cat] += 1
                else:
                    other_classes[cat] = other_classes.get(cat, 0) + 1

                box = veh['box2d']
                x1 = box['x1'] * scale_x
                y1 = box['y1'] * scale_y
                x2 = box['x2'] * scale_x
                y2 = box['y2'] * scale_y

                bw = x2 - x1
                bh = y2 - y1

                if bw <= 0 or bh <= 0:
                    degenerate_boxes += 1
                    continue

                if x1 < 0 or y1 < 0 or x2 > TARGET_W or y2 > TARGET_H:
                    clipped_boxes += 1

                area = bw * bh
                if area < 32 * 32:
                    scale_counts['small (<32^2)'] += 1
                elif area < 96 * 96:
                    scale_counts['medium (32^2 - 96^2)'] += 1
                else:
                    scale_counts['large (>96^2)'] += 1

                # Stride 8 collision check
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                gx = min(grid_w - 1, max(0, int(cx // stride)))
                gy = min(grid_h - 1, max(0, int(cy // stride)))
                cell_occupants_s8.setdefault((gy, gx), []).append(v_idx)

                # Multi-scale routing simulation
                max_dim = max(bw, bh)
                if max_dim < 48:
                    level = 2
                    s = 4
                elif max_dim < 128:
                    level = 3
                    s = 8
                else:
                    level = 4
                    s = 16
                mgx = int(cx // s)
                mgy = int(cy // s)
                cell_occupants_fpn.setdefault((level, mgy, mgx), []).append(v_idx)

            # Analyze collisions in this image
            img_col_s8 = sum(len(vlist) - 1 for vlist in cell_occupants_s8.values() if len(vlist) > 1)
            if img_col_s8 > 0:
                images_with_collision_s8 += 1
                total_overwritten_s8 += img_col_s8

            img_col_fpn = sum(len(vlist) - 1 for vlist in cell_occupants_fpn.values() if len(vlist) > 1)
            if img_col_fpn > 0:
                images_with_collision_fpn += 1
                total_overwritten_fpn += img_col_fpn

        col_rate_s8 = (total_overwritten_s8 / max(1, total_vehicles)) * 100
        col_rate_fpn = (total_overwritten_fpn / max(1, total_vehicles)) * 100

        print(f"Total Images: {total_images} (Found on disk: {existing_images})")
        print(f"Total Ground-Truth Vehicles: {total_vehicles}")
        print(f"Class Breakdown: {class_counts}")
        print(f"Scale Breakdown: {scale_counts}")
        print(f"Degenerate / Inverted Boxes: {degenerate_boxes}")
        print(f"Boxes Touching / Exceeding Boundaries: {clipped_boxes}")
        print(f"\n[CRITICAL FINDING: GRID COLLISIONS]")
        print(f"Single-Scale Stride-8 (64x36):")
        print(f"  - Images with cell collisions: {images_with_collision_s8} / {total_images} ({images_with_collision_s8/total_images*100:.1f}%)")
        print(f"  - Total vehicles overwritten/lost: {total_overwritten_s8} / {total_vehicles} ({col_rate_s8:.2f}%)")
        print(f"Multi-Scale FPN (Strides 4, 8, 16):")
        print(f"  - Images with cell collisions: {images_with_collision_fpn} / {total_images} ({images_with_collision_fpn/total_images*100:.1f}%)")
        print(f"  - Total vehicles overwritten/lost: {total_overwritten_fpn} / {total_vehicles} ({col_rate_fpn:.2f}%)")

        audit_summary[split_name] = {
            'total_images': total_images,
            'existing_images': existing_images,
            'total_vehicles': total_vehicles,
            'class_breakdown': class_counts,
            'scale_breakdown': scale_counts,
            'degenerate_boxes': degenerate_boxes,
            'clipped_boxes': clipped_boxes,
            'single_scale_stride8': {
                'images_with_collisions': images_with_collision_s8,
                'overwritten_vehicles': total_overwritten_s8,
                'collision_loss_percentage': round(col_rate_s8, 2)
            },
            'multiscale_fpn': {
                'images_with_collisions': images_with_collision_fpn,
                'overwritten_vehicles': total_overwritten_fpn,
                'collision_loss_percentage': round(col_rate_fpn, 2)
            }
        }

    # Generate visual validation samples with GT boxes
    print("\n--- Generating Ground-Truth Box Visual Overlays ---")
    val_records = splits['val']
    np.random.seed(42)
    sample_indices = [0, 5, 12, 25, 40]
    palette = {
        'car': (0, 255, 0),         # Green
        'bus': (255, 200, 0),       # Cyan/Yellow
        'truck': (0, 165, 255),     # Orange
        'motorcycle': (0, 0, 255)   # Red
    }

    for i, idx in enumerate(sample_indices):
        if idx >= len(val_records):
            continue
        rec = val_records[idx]
        img_rel = rec['image_path'].replace('\\', '/')
        img_full = os.path.join(BDD_ROOT_DIR, *img_rel.split('/'))
        img = cv2.imread(img_full)
        if img is None:
            continue

        h_orig, w_orig = img.shape[:2]
        img_vis = cv2.resize(img, (TARGET_W, TARGET_H))

        scale_x = TARGET_W / w_orig
        scale_y = TARGET_H / h_orig

        for veh in rec.get('vehicles', []):
            cat = veh.get('category', 'car')
            color = palette.get(cat, (255, 255, 255))
            b = veh['box2d']
            x1 = int(b['x1'] * scale_x)
            y1 = int(b['y1'] * scale_y)
            x2 = int(b['x2'] * scale_x)
            y2 = int(b['y2'] * scale_y)

            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cat}"
            cv2.putText(img_vis, label, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        out_img_path = os.path.join(VISUAL_DIR, f"gt_audit_sample_{i+1}.jpg")
        cv2.imwrite(out_img_path, img_vis)
        print(f"Saved visual sample: {out_img_path}")

    with open(REPORT_JSON_PATH, 'w') as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\nAudit complete. Machine report written to: {REPORT_JSON_PATH}")

if __name__ == '__main__':
    run_audit()
