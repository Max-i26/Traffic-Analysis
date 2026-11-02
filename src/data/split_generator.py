"""
Deterministic, leakage-safe dataset split generator.
Ensures frames belonging to the same recording session / video clip remain in the same split.
"""

import os
import json
import random
from typing import Dict, List, Tuple

def generate_tusimple_pilot_splits(annot_file: str,
                                   pilot_dir: str,
                                   val_ratio: float = 0.2,
                                   seed: int = 42) -> Dict[str, List[Dict]]:
    """
    Groups pilot images by their parent clip and creates train/val splits without leakage.
    """
    random.seed(seed)

    with open(annot_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Filter to only records present in pilot_dir
    available_records = []
    for r in records:
        raw_file = r['raw_file']
        full_img = os.path.join(pilot_dir, raw_file.replace('/', os.sep))
        seg_rel = raw_file.replace('clips/', 'seg_label/').replace('.jpg', '.png')
        full_seg = os.path.join(pilot_dir, seg_rel.replace('/', os.sep))
        if os.path.exists(full_img) and os.path.exists(full_seg):
            available_records.append(r)

    print(f"Total verified pilot records on disk: {len(available_records)}")

    # Group by clip: clips/0313-1/6040/20.jpg -> clip_id: "0313-1/6040"
    clip_groups: Dict[str, List[Dict]] = {}
    for r in available_records:
        parts = r['raw_file'].split('/')
        if len(parts) >= 3:
            clip_id = f"{parts[1]}/{parts[2]}"
        else:
            clip_id = r['raw_file']
        clip_groups.setdefault(clip_id, []).append(r)

    unique_clips = list(clip_groups.keys())
    random.shuffle(unique_clips)

    val_clip_count = max(1, int(len(unique_clips) * val_ratio))
    val_clips = set(unique_clips[:val_clip_count])
    train_clips = set(unique_clips[val_clip_count:])

    train_records = []
    val_records = []

    for cid in train_clips:
        train_records.extend(clip_groups[cid])
    for cid in val_clips:
        val_records.extend(clip_groups[cid])

    splits = {
        'metadata': {
            'seed': seed,
            'val_ratio': val_ratio,
            'total_samples': len(available_records),
            'train_samples': len(train_records),
            'val_samples': len(val_records),
            'train_clips': len(train_clips),
            'val_clips': len(val_clips)
        },
        'train': train_records,
        'val': val_records
    }

    return splits

if __name__ == '__main__':
    annot = r'datasets/annotations/tusimple/label_data_0313.json'
    pilot = r'datasets/extracted/tusimple_pilot'
    out_split = r'datasets/splits/tusimple_pilot_splits.json'

    splits = generate_tusimple_pilot_splits(annot, pilot)
    os.makedirs(os.path.dirname(out_split), exist_ok=True)
    with open(out_split, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)

    print(f"Splits saved to {out_split}:")
    print(f"  Train: {splits['metadata']['train_samples']} frames ({splits['metadata']['train_clips']} clips)")
    print(f"  Val:   {splits['metadata']['val_samples']} frames ({splits['metadata']['val_clips']} clips)")
