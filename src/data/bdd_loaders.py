"""
PyTorch Datasets and DataLoaders for BDD100K Road Segmentation and Vehicle Detection.
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, List

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VEHICLE_MAP = {
    'car': 0,
    'bus': 1,
    'truck': 2,
    'motorcycle': 3
}

class BDDDrivableDataset(Dataset):
    """Dataset for Module C: Drivable road area surface segmentation."""
    def __init__(self, records: List[Dict], root_dir: str, target_size: Tuple[int, int] = (512, 288)):
        self.records = records
        self.root_dir = root_dir
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img_rel = rec['image_path'].replace('\\', '/')
        mask_rel = rec['mask_path'].replace('\\', '/')
        img_path = os.path.join(self.root_dir, *img_rel.split('/'))
        mask_path = os.path.join(self.root_dir, *mask_rel.split('/'))

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f"Missing file for {rec['name']} at {img_path}")

        target_w, target_h = self.target_size
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # Drivable area mask (value 1 in BDD masks is drivable)
        drivable_binary = (mask_resized > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(drivable_binary).unsqueeze(0).float()

        return img_tensor, mask_tensor

class BDDVehicleDetectionDataset(Dataset):
    """Dataset for Module B: Anchor-Free Multi-Scale Vehicle Detection."""
    def __init__(self, records: List[Dict], root_dir: str, target_size: Tuple[int, int] = (512, 288), stride: int = 8):
        self.records = records
        self.root_dir = root_dir
        self.target_size = target_size # (W, H)
        self.stride = stride
        self.grid_w = target_size[0] // stride # 64
        self.grid_h = target_size[1] // stride # 36

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img_rel = rec['image_path'].replace('\\', '/')
        img_path = os.path.join(self.root_dir, *img_rel.split('/'))
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Missing file for {rec['name']} at {img_path}")

        orig_h, orig_w = rec['height'], rec['width']
        target_w, target_h = self.target_size

        # Resize image & normalize
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # Target grids
        obj_target = np.zeros((1, self.grid_h, self.grid_w), dtype=np.float32)
        cls_target = np.zeros((4, self.grid_h, self.grid_w), dtype=np.float32)
        box_target = np.zeros((4, self.grid_h, self.grid_w), dtype=np.float32)

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        for veh in rec['vehicles']:
            cat = veh['category']
            if cat not in VEHICLE_MAP:
                continue
            cat_id = VEHICLE_MAP[cat]
            b = veh['box2d']

            # Scaled box in resized image coords
            x1 = max(0.0, b['x1'] * scale_x)
            y1 = max(0.0, b['y1'] * scale_y)
            x2 = min(float(target_w), b['x2'] * scale_x)
            y2 = min(float(target_h), b['y2'] * scale_y)

            bw = x2 - x1
            bh = y2 - y1
            if bw < 4 or bh < 4:
                continue

            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0

            # Grid cell assignment
            gx = min(self.grid_w - 1, int(cx // self.stride))
            gy = min(self.grid_h - 1, int(cy // self.stride))

            obj_target[0, gy, gx] = 1.0
            cls_target[cat_id, gy, gx] = 1.0
            # Normalized box representation: [cx/W, cy/H, w/W, h/H]
            box_target[:, gy, gx] = [cx / target_w, cy / target_h, bw / target_w, bh / target_h]

        return (
            img_tensor,
            torch.from_numpy(obj_target),
            torch.from_numpy(cls_target),
            torch.from_numpy(box_target)
        )

def get_bdd_dataloaders(task: str = 'drivable',
                        split_file: str = 'datasets/splits/bdd_pilot_splits.json',
                        root_dir: str = 'datasets/extracted/bdd_pilot',
                        batch_size: int = 4,
                        target_size: Tuple[int, int] = (512, 288),
                        num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    with open(split_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    if task == 'drivable':
        train_ds = BDDDrivableDataset(splits['train'], root_dir, target_size)
        val_ds = BDDDrivableDataset(splits['val'], root_dir, target_size)
    elif task == 'detection':
        train_ds = BDDVehicleDetectionDataset(splits['train'], root_dir, target_size)
        val_ds = BDDVehicleDetectionDataset(splits['val'], root_dir, target_size)
    else:
        raise ValueError(f"Unknown task {task}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader
