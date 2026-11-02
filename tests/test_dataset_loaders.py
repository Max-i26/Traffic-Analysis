"""
Unit and integration tests for data engineering and PyTorch DataLoaders.
Verifies shapes, tensor types, and GPU allocation on RTX 2050.
"""

import os
import unittest
import torch
import numpy as np

from src.data.annotation_parser import parse_tusimple_record
from src.data.preprocessing import rasterize_lane_mask, preprocess_image_and_mask
from src.data.dataset_loaders import get_tusimple_dataloaders

class TestDataEngineering(unittest.TestCase):

    def test_tusimple_parser_discards_negative_samples(self):
        sample_record = {
            'raw_file': 'clips/test/001/20.jpg',
            'lanes': [
                [-2, -2, 500, 520, 540], # valid points: 3
                [-2, -2, -2, -2, -2]     # valid points: 0 (should be discarded)
            ],
            'h_samples': [200, 220, 240, 260, 280]
        }
        parsed = parse_tusimple_record(sample_record)
        self.assertEqual(parsed['num_lanes'], 1)
        self.assertEqual(len(parsed['lane_points'][0]), 3)
        self.assertEqual(parsed['lane_points'][0][0], (500, 240))

    def test_rasterize_mask_dimensions_and_values(self):
        lane_points = [[(100, 200), (200, 400), (300, 600)]]
        mask = rasterize_lane_mask(lane_points, height=720, width=1280, thickness=6)
        self.assertEqual(mask.shape, (720, 1280))
        self.assertEqual(mask.dtype, np.uint8)
        unique_vals = set(np.unique(mask))
        self.assertTrue(unique_vals.issubset({0, 1}))
        self.assertGreater(mask.sum(), 0)

    def test_dataloader_batch_shapes_and_gpu_transfer(self):
        split_file = 'datasets/splits/tusimple_pilot_splits.json'
        pilot_dir = 'datasets/extracted/tusimple_pilot'

        if not os.path.exists(split_file) or not os.path.exists(pilot_dir):
            self.skipTest("Pilot dataset or splits not generated yet.")

        batch_size = 4
        target_size = (512, 288)
        train_loader, val_loader = get_tusimple_dataloaders(
            split_file=split_file,
            pilot_dir=pilot_dir,
            batch_size=batch_size,
            target_size=target_size,
            num_workers=0 # use 0 for quick test execution
        )

        # Fetch one batch
        images, masks, info = next(iter(train_loader))

        # Check tensor shapes: [Batch, Channels, Height, Width]
        self.assertEqual(images.shape, (batch_size, 3, 288, 512))
        self.assertEqual(masks.shape, (batch_size, 1, 288, 512))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(masks.dtype, torch.float32)

        # Check physical GPU transfer if CUDA available
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            images_gpu = images.to(device)
            masks_gpu = masks.to(device)
            self.assertEqual(images_gpu.device.type, 'cuda')
            self.assertEqual(masks_gpu.device.type, 'cuda')
            allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
            print(f"\n[GPU Physical Test] Batch of 4 tensors loaded to RTX 2050!")
            print(f"  Images Tensor: {images_gpu.shape} ({images_gpu.dtype})")
            print(f"  Masks Tensor:  {masks_gpu.shape} ({masks_gpu.dtype})")
            print(f"  VRAM Allocated: {allocated_mb:.2f} MB")

if __name__ == '__main__':
    unittest.main()
