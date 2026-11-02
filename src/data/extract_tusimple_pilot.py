import zipfile
import os
import json

zip_path = r'E:\Neural_Network_project\Data\archive (1).zip'
output_annot_dir = r'datasets/annotations/tusimple'
output_pilot_dir = r'datasets/extracted/tusimple_pilot'

os.makedirs(output_annot_dir, exist_ok=True)
os.makedirs(output_pilot_dir, exist_ok=True)

print(f"Opening TuSimple archive: {zip_path}")
with zipfile.ZipFile(zip_path, 'r') as zf:
    # 1. Extract JSON annotation files
    json_targets = [
        'TUSimple/train_set/label_data_0313.json',
        'TUSimple/train_set/label_data_0531.json',
        'TUSimple/train_set/label_data_0601.json',
        'TUSimple/test_label.json'
    ]
    for target in json_targets:
        basename = os.path.basename(target)
        dest_path = os.path.join(output_annot_dir, basename)
        print(f"Extracting {target} -> {dest_path}")
        with zf.open(target) as src_f, open(dest_path, 'wb') as dst_f:
            dst_f.write(src_f.read())

    # 2. Extract a pilot set of 200 annotated frames + corresponding seg_labels
    annot_file = os.path.join(output_annot_dir, 'label_data_0313.json')
    with open(annot_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(lines)} annotations from label_data_0313.json")
    pilot_count = 200
    extracted_pilot = 0

    for i in range(min(pilot_count, len(lines))):
        record = json.loads(lines[i])
        raw_file = record['raw_file'] # e.g. "clips/0313-1/6040/20.jpg"
        
        # In zip archive, the path is TUSimple/train_set/clips/0313-1/6040/20.jpg
        zip_img_path = f"TUSimple/train_set/{raw_file}"
        dest_img_path = os.path.join(output_pilot_dir, raw_file.replace('/', os.sep))
        
        # Corresponding seg_label is seg_label/0313-1/6040/20.png
        seg_rel = raw_file.replace('clips/', 'seg_label/').replace('.jpg', '.png')
        zip_seg_path = f"TUSimple/train_set/{seg_rel}"
        dest_seg_path = os.path.join(output_pilot_dir, seg_rel.replace('/', os.sep))

        os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(dest_seg_path), exist_ok=True)

        # Extract image if exists
        try:
            with zf.open(zip_img_path) as s, open(dest_img_path, 'wb') as d:
                d.write(s.read())
            with zf.open(zip_seg_path) as s, open(dest_seg_path, 'wb') as d:
                d.write(s.read())
            extracted_pilot += 1
        except KeyError as e:
            print(f"Skipping {raw_file}: not found in zip ({e})")

    print(f"Successfully extracted {extracted_pilot} pilot image pairs (image + mask) into {output_pilot_dir}")
