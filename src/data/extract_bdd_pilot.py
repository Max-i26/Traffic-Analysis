"""
BDD100K Pilot Extractor and Split Generator for Phase 4.
Extracts paired images, drivable area ground-truth masks, and vehicle detection annotations.
"""

import os
import json
import zipfile

def extract_bdd_pilot(count: int = 250):
    bdd_annot_file = 'datasets/annotations/bdd_detections.json'
    img_zip_path = r'E:\Neural_Network_project\Data\bdd100k_images_100k.zip'
    drv_zip_path = r'E:\Neural_Network_project\Data\bdd100k_drivable_maps.zip'
    output_dir = 'datasets/extracted/bdd_pilot'
    split_file = 'datasets/splits/bdd_pilot_splits.json'

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'drivable_masks'), exist_ok=True)
    os.makedirs(os.path.dirname(split_file), exist_ok=True)

    print(f"Loading annotations from {bdd_annot_file}...")
    with open(bdd_annot_file, 'r', encoding='utf-8') as f:
        bdd_data = json.load(f)

    zf_img = zipfile.ZipFile(img_zip_path, 'r')
    zf_drv = zipfile.ZipFile(drv_zip_path, 'r')

    # Index zip entries by basename
    print("Indexing archives...")
    img_map = {os.path.basename(n): n for n in zf_img.namelist() if n.endswith('.jpg')}
    drv_map = {os.path.basename(n).replace('_drivable_id.png', '.jpg'): n for n in zf_drv.namelist() if n.endswith('_drivable_id.png')}

    vehicle_classes = {'car', 'bus', 'truck', 'motorcycle'}
    extracted_records = []

    print(f"Extracting {count} paired samples...")
    for frame in bdd_data['frames']:
        if len(extracted_records) >= count:
            break

        name = frame['name'] # e.g. "b1c66a42-6f7d68ca.jpg"
        if name in img_map and name in drv_map:
            # Filter labels to vehicle classes
            vehicles = []
            for lbl in frame['labels']:
                if lbl['category'] in vehicle_classes:
                    vehicles.append(lbl)

            # Require at least one vehicle in the frame
            if not vehicles:
                continue

            # Extract image
            img_dest = os.path.join(output_dir, 'images', name)
            with zf_img.open(img_map[name]) as s, open(img_dest, 'wb') as d:
                d.write(s.read())

            # Extract drivable mask
            mask_dest = os.path.join(output_dir, 'drivable_masks', name.replace('.jpg', '_drivable.png'))
            with zf_drv.open(drv_map[name]) as s, open(mask_dest, 'wb') as d:
                d.write(s.read())

            extracted_records.append({
                'name': name,
                'image_path': f"images/{name}",
                'mask_path': f"drivable_masks/{name.replace('.jpg', '_drivable.png')}",
                'width': frame['width'],
                'height': frame['height'],
                'vehicles': vehicles
            })

    print(f"Successfully extracted {len(extracted_records)} paired frames!")

    # 80/20 train/val split
    val_count = int(len(extracted_records) * 0.2)
    val_records = extracted_records[:val_count]
    train_records = extracted_records[val_count:]

    splits = {
        'metadata': {
            'total_samples': len(extracted_records),
            'train_samples': len(train_records),
            'val_samples': len(val_records),
            'vehicle_classes': sorted(list(vehicle_classes))
        },
        'train': train_records,
        'val': val_records
    }

    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)

    print(f"Splits saved to {split_file}:")
    print(f"  - Train: {len(train_records)} frames")
    print(f"  - Val:   {len(val_records)} frames")

if __name__ == '__main__':
    extract_bdd_pilot(count=250)
