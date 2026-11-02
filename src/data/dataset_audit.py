import os
import json
import cv2
import numpy as np
import zipfile

def audit_tusimple(annot_file, pilot_dir, report_dir):
    print("--- Auditing TuSimple Lane Annotations ---")
    with open(annot_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    total_records = len(lines)
    invalid_coord_count = 0
    total_lanes = 0
    valid_points_count = 0
    lanes_per_image = []

    for line in lines:
        record = json.loads(line)
        lanes = record.get('lanes', [])
        h_samples = record.get('h_samples', [])
        total_lanes += len(lanes)
        lanes_per_image.append(len(lanes))

        for lane in lanes:
            for x in lane:
                if x == -2:
                    invalid_coord_count += 1
                else:
                    valid_points_count += 1

    print(f"Total Annotated Frames: {total_records}")
    print(f"Total Lane Objects: {total_lanes} (Avg: {np.mean(lanes_per_image):.2f} lanes/frame)")
    print(f"Valid Coordinates: {valid_points_count:,}")
    print(f"Invalid / Occluded (-2) Coordinates: {invalid_coord_count:,}")

    # Visual overlay sample generation from pilot
    os.makedirs(report_dir, exist_ok=True)
    visual_sample_count = 0
    
    # Iterate through first 5 pilot records to generate visual samples
    for i in range(min(5, len(lines))):
        record = json.loads(lines[i])
        raw_file = record['raw_file']
        img_path = os.path.join(pilot_dir, raw_file.replace('/', os.sep))
        seg_rel = raw_file.replace('clips/', 'seg_label/').replace('.jpg', '.png')
        seg_path = os.path.join(pilot_dir, seg_rel.replace('/', os.sep))

        if os.path.exists(img_path) and os.path.exists(seg_path):
            img = cv2.imread(img_path)
            mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            
            # Create overlay
            overlay = img.copy()
            # Draw ground truth lane lines from coordinates
            h_samples = record['h_samples']
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)]
            
            for lane_idx, lane in enumerate(record['lanes']):
                pts = []
                for x, y in zip(lane, h_samples):
                    if x >= 0:
                        pts.append((int(x), int(y)))
                if len(pts) > 1:
                    cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, colors[lane_idx % len(colors)], 3)

            # Draw segmentation mask in semi-transparent color
            color_mask = np.zeros_like(img)
            color_mask[mask > 0] = [0, 165, 255] # orange
            blended = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)

            # Add HUD text
            cv2.putText(blended, f"TuSimple Ground Truth - Frame {i+1}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(blended, f"Lanes: {len(record['lanes'])} | Seg Mask: Present", (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out_path = os.path.join(report_dir, f"tusimple_sample_{i+1}.jpg")
            cv2.imwrite(out_path, blended)
            visual_sample_count += 1

    print(f"Generated {visual_sample_count} TuSimple visual verification samples in {report_dir}")
    return {
        'total_records': total_records,
        'total_lanes': total_lanes,
        'avg_lanes': float(np.mean(lanes_per_image)),
        'valid_points': valid_points_count,
        'invalid_points': invalid_coord_count
    }

def audit_bdd(bdd_file, img_zip_path, report_dir):
    print("\n--- Auditing BDD100K Detection Annotations ---")
    with open(bdd_file, 'r', encoding='utf-8') as f:
        bdd = json.load(f)

    total_frames = bdd['total_frames']
    category_counts = bdd['category_counts']
    print(f"Total Annotated Frames: {total_frames}")
    print(f"Total Categories: {len(category_counts)}")

    # Check image matching in bdd100k_images_100k.zip
    print(f"Verifying images in {img_zip_path}...")
    zf = zipfile.ZipFile(img_zip_path, 'r')
    zip_names = set(os.path.basename(n) for n in zf.namelist() if n.endswith('.jpg'))

    matched_images = 0
    for frame in bdd['frames'][:500]: # check first 500
        if frame['name'] in zip_names:
            matched_images += 1
    match_pct = (matched_images / 500.0) * 100.0
    print(f"Sample Image Match Rate: {matched_images}/500 ({match_pct:.1f}%)")

    # Generate visual sample with bounding boxes
    os.makedirs(report_dir, exist_ok=True)
    samples_created = 0

    # Color palette for classes
    palette = {
        'car': (0, 255, 0),
        'bus': (255, 100, 0),
        'truck': (200, 200, 0),
        'motorcycle': (0, 165, 255),
        'pedestrian': (0, 0, 255),
        'bicycle': (255, 0, 255),
        'traffic light': (255, 255, 0),
        'traffic sign': (0, 255, 255)
    }

    # Find 3 frames that exist in the zip and draw boxes
    for frame in bdd['frames']:
        if samples_created >= 3:
            break
        # Locate in zip
        matching_paths = [n for n in zf.namelist() if n.endswith(frame['name'])]
        if matching_paths:
            img_bytes = zf.read(matching_paths[0])
            img_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            for lbl in frame['labels']:
                cat = lbl['category']
                color = palette.get(cat, (255, 255, 255))
                box = lbl['box2d']
                x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # Text badge
                cv2.putText(img, cat, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.putText(img, f"BDD100K BBoxes - {frame['name']}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            out_path = os.path.join(report_dir, f"bdd_sample_{samples_created+1}.jpg")
            cv2.imwrite(out_path, img)
            samples_created += 1

    print(f"Generated {samples_created} BDD100K visual verification samples in {report_dir}")
    return {
        'total_frames': total_frames,
        'category_counts': category_counts,
        'match_pct': match_pct
    }

if __name__ == '__main__':
    tusimple_file = r'datasets/annotations/tusimple/label_data_0313.json'
    pilot_dir = r'datasets/extracted/tusimple_pilot'
    bdd_file = r'datasets/annotations/bdd_detections.json'
    bdd_img_zip = r'E:\Neural_Network_project\Data\bdd100k_images_100k.zip'
    report_dir = r'reports/visual_samples'

    res_tu = audit_tusimple(tusimple_file, pilot_dir, report_dir)
    res_bdd = audit_bdd(bdd_file, bdd_img_zip, report_dir)
    print("\nAudit completed successfully!")
