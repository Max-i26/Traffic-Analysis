import json
import os

src_file = r'datasets/annotations/bdd_fiftyone_samples.json'
dest_file = r'datasets/annotations/bdd_detections.json'

print(f"Loading {src_file}...")
with open(src_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

samples = data.get('samples', [])
print(f"Total samples found in FiftyOne export: {len(samples)}")

standard_records = []
category_counts = {}

for s in samples:
    raw_filepath = s.get('filepath', '')
    img_name = os.path.basename(raw_filepath) # e.g. "b1c66a42-6f7d68ca.jpg"
    meta = s.get('metadata', {})
    width = meta.get('width', 1280)
    height = meta.get('height', 720)

    labels = []
    dets = s.get('detections', {}).get('detections', [])
    for d in dets:
        cat = d.get('label')
        category_counts[cat] = category_counts.get(cat, 0) + 1
        bbox = d.get('bounding_box', [0, 0, 0, 0]) # [x, y, w, h] normalized
        x1 = round(bbox[0] * width, 2)
        y1 = round(bbox[1] * height, 2)
        x2 = round((bbox[0] + bbox[2]) * width, 2)
        y2 = round((bbox[1] + bbox[3]) * height, 2)
        
        labels.append({
            'category': cat,
            'box2d': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'occluded': d.get('occluded', False),
            'truncated': d.get('truncated', False)
        })

    standard_records.append({
        'name': img_name,
        'width': width,
        'height': height,
        'labels': labels
    })

output_data = {
    'total_frames': len(standard_records),
    'categories': sorted(list(category_counts.keys())),
    'category_counts': category_counts,
    'frames': standard_records
}

print("Category breakdown across 20,001 frames:")
for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
    print(f"  - {cat:15s}: {count:,} instances")

print(f"Writing standardized annotations to {dest_file}...")
with open(dest_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2)

print(f"Done! Saved {os.path.getsize(dest_file) / 1e6:.1f} MB.")
