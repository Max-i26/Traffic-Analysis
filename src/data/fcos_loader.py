"""
FCOS Dataset Loader with Class-Aware Balanced Sampling and Spatial Augmentations.
Provides clean batching of images and variable-length ground-truth bounding box lists.
"""

import os
import json
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Dict, Tuple, Optional

CLASS_NAMES = ['car', 'bus', 'truck', 'motorcycle']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

def fcos_collate_fn(batch):
    """
    Collate function for variable-length bounding boxes.
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return imgs, targets

class BDD100kFCOSDataset(Dataset):
    def __init__(self,
                 records: List[Dict],
                 root_dir: str,
                 target_size: Tuple[int, int] = (512, 288),
                 augment: bool = False):
        self.records = records
        self.root_dir = root_dir
        self.target_w, self.target_h = target_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = self.records[idx]
        img_rel = rec['image_path'].replace('\\', '/')
        img_path = os.path.join(self.root_dir, *img_rel.split('/'))

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Missing image file: {img_path}")

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (self.target_w, self.target_h))

        scale_x = self.target_w / orig_w
        scale_y = self.target_h / orig_h

        boxes = []
        labels = []

        for veh in rec.get('vehicles', []):
            cat = veh.get('category', 'car')
            if cat not in CLASS_TO_ID:
                continue
            cat_id = CLASS_TO_ID[cat]
            b = veh['box2d']

            x1 = b['x1'] * scale_x
            y1 = b['y1'] * scale_y
            x2 = b['x2'] * scale_x
            y2 = b['y2'] * scale_y

            # Clamp to image boundaries
            x1 = max(0.0, min(float(self.target_w), x1))
            y1 = max(0.0, min(float(self.target_h), y1))
            x2 = max(0.0, min(float(self.target_w), x2))
            y2 = max(0.0, min(float(self.target_h), y2))

            if (x2 - x1) > 2.0 and (y2 - y1) > 2.0:
                boxes.append([x1, y1, x2, y2])
                labels.append(cat_id)

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.empty((0,), dtype=np.int64)

        # Data Augmentations during training
        if self.augment:
            # 1. Random Horizontal Flip (50% probability)
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                if len(boxes) > 0:
                    old_x1 = boxes[:, 0].copy()
                    old_x2 = boxes[:, 2].copy()
                    boxes[:, 0] = self.target_w - old_x2
                    boxes[:, 2] = self.target_w - old_x1

            # 2. Photometric jitter (brightness / contrast)
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15) # Contrast
                beta = random.uniform(-15, 15)      # Brightness
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Normalize with ImageNet stats
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_rgb - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        target = {
            'boxes': torch.from_numpy(boxes).float(),
            'labels': torch.from_numpy(labels).long(),
            'image_id': idx
        }

        return img_tensor, target

def get_fcos_dataloaders(split_file: str = 'datasets/splits/bdd_pilot_splits.json',
                         root_dir: str = 'datasets/extracted/bdd_pilot',
                         batch_size: int = 4,
                         target_size: Tuple[int, int] = (512, 288),
                         num_workers: int = 0,
                         use_class_aware_sampling: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and val DataLoaders for FCOS.
    Applies class-aware sampling so minority classes (bus, truck, motorcycle) appear proportionally more often.
    """
    with open(split_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    train_recs = splits['train']
    val_recs = splits['val']

    train_ds = BDD100kFCOSDataset(train_recs, root_dir, target_size=target_size, augment=True)
    val_ds = BDD100kFCOSDataset(val_recs, root_dir, target_size=target_size, augment=False)

    sampler = None
    if use_class_aware_sampling:
        # Calculate weights per sample
        weights = []
        for rec in train_recs:
            cats = [v.get('category') for v in rec.get('vehicles', [])]
            w = 1.0
            if 'motorcycle' in cats:
                w += 10.0 # Rare class boost
            if 'bus' in cats:
                w += 4.0
            if 'truck' in cats:
                w += 2.5
            weights.append(w)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=fcos_collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=fcos_collate_fn,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
