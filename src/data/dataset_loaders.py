"""
PyTorch Dataset and DataLoader implementation for TuSimple Lane Detection.
Optimized for memory efficiency and RTX 2050 4GB VRAM constraints.
"""

import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Optional

from src.data.annotation_parser import parse_tusimple_record
from src.data.preprocessing import preprocess_image_and_mask, rasterize_lane_mask

class TuSimpleLaneDataset(Dataset):
    """
    PyTorch Dataset for TuSimple lane detection and segmentation.
    Loads raw road images and ground truth segmentation masks.
    """
    def __init__(self,
                 records: list,
                 pilot_dir: str,
                 target_size: Tuple[int, int] = (512, 288),
                 use_precomputed_masks: bool = True):
        self.records = records
        self.pilot_dir = pilot_dir
        self.target_size = target_size
        self.use_precomputed_masks = use_precomputed_masks

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.records[idx]
        raw_file = record['raw_file']
        img_path = os.path.join(self.pilot_dir, raw_file.replace('/', os.sep))

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        # Get mask: either load seg_label PNG or rasterize dynamically
        mask = None
        if self.use_precomputed_masks:
            seg_rel = raw_file.replace('clips/', 'seg_label/').replace('.jpg', '.png')
            seg_path = os.path.join(self.pilot_dir, seg_rel.replace('/', os.sep))
            if os.path.exists(seg_path):
                raw_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                mask = (raw_mask > 0).astype('uint8')

        if mask is None:
            # Fallback to rasterizing from coordinates
            parsed = parse_tusimple_record(record)
            h, w = img.shape[:2]
            mask = rasterize_lane_mask(parsed['lane_points'], height=h, width=w)

        # Preprocess, resize to target_size (512, 288), and convert to tensors
        img_tensor, mask_tensor = preprocess_image_and_mask(img, mask, target_size=self.target_size)

        info = {
            'raw_file': raw_file,
            'orig_size': (img.shape[1], img.shape[0]) # (width, height)
        }

        return img_tensor, mask_tensor, info

def get_tusimple_dataloaders(split_file: str,
                             pilot_dir: str,
                             batch_size: int = 4,
                             target_size: Tuple[int, int] = (512, 288),
                             num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for training and validation splits.
    Configured with pin_memory=True for fast CUDA transfers to the RTX 2050 GPU.
    """
    with open(split_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    train_dataset = TuSimpleLaneDataset(
        records=splits['train'],
        pilot_dir=pilot_dir,
        target_size=target_size
    )

    val_dataset = TuSimpleLaneDataset(
        records=splits['val'],
        pilot_dir=pilot_dir,
        target_size=target_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
