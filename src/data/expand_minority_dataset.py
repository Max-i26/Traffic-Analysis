"""
Extracts a targeted minority-class expansion (trucks, buses, motorcycles) from BDD100K zip.
Expands the training set from 200 to 450 images while preserving the validation split 100% untouched.
Generates: datasets/splits/bdd_expanded_splits.json
"""

import os
import sys
import json
import zipfile
from typing import Dict, List, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

ZIP_PATH = r'E:\Neural_Network_project\Data\bdd100k_images_100k.zip'
BDD_ANNOTS_PATH = r'datasets/annotations/bdd_detections.json'
ORIGINAL_SPLITS_PATH = r'datasets/splits/bdd_pilot_splits.json'
OUTPUT_SPLITS_PATH = r'datasets/splits/bdd_expanded_splits.json'
EXTRACT_IMG_DIR = r'datasets/extracted/bdd_pilot/images'

TARGET_CLASSES = {'car', 'bus', 'truck', 'motorcycle'}

def expand_minority_dataset():
    print("=" * 70)
    print("STAGE 5: MINORITY-CLASS DATASET EXPANSION (TRUCKS, BUSES, MOTORCYCLES)")
    print("=" * 70)

    # 1. Load original splits to ensure validation set remains 100% untouched
    with open(ORIGINAL_SPLITS_PATH, 'r') as f:
        orig_splits = json.load(f)

    existing_train = orig_splits['train']
    val_set = orig_splits['val']

    existing_names: Set[str] = {rec['name'] for rec in existing_train + val_set}
    print(f"Existing pilot: {len(existing_train)} train, {len(val_set)} val images.")

    # 2. Ingest BDD annotations
    print(f"Reading BDD detections from: {BDD_ANNOTS_PATH}")
    with open(BDD_ANNOTS_PATH, 'r') as f:
        bdd_data = json.load(f)
    frames = bdd_data.get('frames', [])

    # Index candidates by minority classes
    truck_candidates = []
    bus_candidates = []
    motorcycle_candidates = []

    for f in frames:
        name = f['name']
        if name in existing_names:
            continue

        cats = {l['category'] for l in f.get('labels', []) if l.get('category') in TARGET_CLASSES}
        if 'motorcycle' in cats:
            motorcycle_candidates.append(f)
        elif 'bus' in cats:
            bus_candidates.append(f)
        elif 'truck' in cats:
            truck_candidates.append(f)

    print(f"Available candidates: Trucks: {len(truck_candidates)}, Buses: {len(bus_candidates)}, Motorcycles: {len(motorcycle_candidates)}")

    # Target selection: 50 motorcycles, 100 buses, 100 trucks
    selected_frames: List[Dict] = []
    selected_frames.extend(motorcycle_candidates[:50])
    selected_frames.extend(bus_candidates[:100])
    selected_frames.extend(truck_candidates[:100])

    print(f"Selected {len(selected_frames)} targeted minority-rich frames for training expansion.")

    # 3. Extract selected frames from zip
    os.makedirs(EXTRACT_IMG_DIR, exist_ok=True)
    new_train_records = []

    print(f"Opening archive: {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zip_namelist = set(zf.namelist())

        for f in selected_frames:
            img_name = f['name']
            # Target path in zip: 100k/train/<name> or 100k/val/<name>
            zip_target = None
            for prefix in ['100k/train/', '100k/val/', '100k/test/']:
                cand = prefix + img_name
                if cand in zip_namelist:
                    zip_target = cand
                    break

            if zip_target is None:
                continue

            dest_path = os.path.join(EXTRACT_IMG_DIR, img_name)
            if not os.path.exists(dest_path):
                with zf.open(zip_target) as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())

            # Convert labels to vehicle record
            vehicles = []
            for l in f.get('labels', []):
                cat = l.get('category')
                if cat in TARGET_CLASSES and 'box2d' in l:
                    vehicles.append({
                        'category': cat,
                        'box2d': l['box2d'],
                        'occluded': l.get('attributes', {}).get('occluded', False),
                        'truncated': l.get('attributes', {}).get('truncated', False)
                    })

            new_train_records.append({
                'name': img_name,
                'image_path': f"images/{img_name}",
                'mask_path': "",
                'width': f.get('width', 1280),
                'height': f.get('height', 720),
                'vehicles': vehicles
            })

    print(f"Extracted and formatted {len(new_train_records)} images.")

    # 4. Create expanded splits
    combined_train = existing_train + new_train_records
    expanded_splits = {
        'train': combined_train,
        'val': val_set # 100% frozen and identical
    }

    with open(OUTPUT_SPLITS_PATH, 'w') as f:
        json.dump(expanded_splits, f, indent=2)

    # Class breakdown in expanded train
    class_counts = {c: 0 for c in TARGET_CLASSES}
    for r in combined_train:
        for v in r.get('vehicles', []):
            cat = v.get('category')
            if cat in class_counts:
                class_counts[cat] += 1

    print("\n" + "=" * 70)
    print("EXPANDED DATASET SUMMARY:")
    print(f"Train Images: {len(combined_train)} (Original: {len(existing_train)}, Added: {len(new_train_records)})")
    print(f"Val Images:   {len(val_set)} (100% Identical Held-Out Split)")
    print(f"Expanded Train Class Instances: {class_counts}")
    print(f"Splits saved to: {OUTPUT_SPLITS_PATH}")
    print("=" * 70)

if __name__ == '__main__':
    expand_minority_dataset()
