"""
Image and mask preprocessing pipeline.
Includes coordinate-to-mask rasterization, resizing, and PyTorch tensor normalization.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional

# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def rasterize_lane_mask(lane_points: List[List[Tuple[int, int]]],
                        height: int = 720,
                        width: int = 1280,
                        thickness: int = 6) -> np.ndarray:
    """
    Renders continuous lane polylines into a binary single-channel mask.
    Returns:
        mask (np.ndarray): uint8 array of shape (height, width) with values 0 or 1.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for pts in lane_points:
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(mask, [pts_arr], isClosed=False, color=1, thickness=thickness)
    return mask

def preprocess_image_and_mask(image: np.ndarray,
                              mask: np.ndarray,
                              target_size: Tuple[int, int] = (512, 288)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares raw image and mask for neural network input:
    1. Resizes image to target (width, height) using bilinear interpolation.
    2. Resizes mask to target (width, height) using nearest-neighbor interpolation.
    3. Normalizes image pixels to standard ImageNet mean and std.
    4. Converts both to PyTorch tensors with shape [C, H, W].

    Args:
        image: np.ndarray of shape (H, W, 3), BGR or RGB uint8
        mask: np.ndarray of shape (H, W), uint8 binary (0, 1)
        target_size: (width, height), default (512, 288) for RTX 2050 4GB

    Returns:
        img_tensor: torch.Tensor of shape [3, target_h, target_w], dtype float32
        mask_tensor: torch.Tensor of shape [1, target_h, target_w], dtype float32
    """
    target_w, target_h = target_size

    # Resize
    resized_img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Convert BGR to RGB if needed, scale to [0, 1]
    if resized_img.ndim == 3 and resized_img.shape[2] == 3:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    img_float = resized_img.astype(np.float32) / 255.0
    img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD

    # Transpose to [C, H, W]
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

    # Mask tensor [1, H, W]
    mask_binary = (resized_mask > 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

    return img_tensor, mask_tensor
