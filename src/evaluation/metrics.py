"""
Evaluation metrics for lane segmentation models.
Computes IoU, Dice Coefficient, Precision, Recall, and Pixel Accuracy.
"""

import torch
import numpy as np
from typing import Dict

def compute_batch_metrics(logits: torch.Tensor,
                          targets: torch.Tensor,
                          threshold: float = 0.5,
                          smooth: float = 1e-6) -> Dict[str, float]:
    """
    Computes binary segmentation metrics over a batch.
    Args:
        logits: Raw network predictions [B, 1, H, W]
        targets: Ground truth binary masks [B, 1, H, W]
        threshold: Classification probability threshold (default 0.5)

    Returns:
        Dict with 'iou', 'dice', 'precision', 'recall', 'pixel_acc'
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        targets = (targets > 0.5).float()

        # Flatten [B, -1]
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (preds_flat * targets_flat).sum(dim=1)
        pred_sum = preds_flat.sum(dim=1)
        target_sum = targets_flat.sum(dim=1)
        union = pred_sum + target_sum - intersection

        # IoU
        iou = ((intersection + smooth) / (union + smooth)).mean().item()

        # Dice / F1
        dice = ((2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)).mean().item()

        # Precision & Recall
        precision = ((intersection + smooth) / (pred_sum + smooth)).mean().item()
        recall = ((intersection + smooth) / (target_sum + smooth)).mean().item()

        # Pixel Accuracy
        correct_pixels = (preds_flat == targets_flat).float().sum(dim=1)
        total_pixels = preds_flat.size(1)
        pixel_acc = (correct_pixels / total_pixels).mean().item()

        return {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'pixel_acc': float(pixel_acc)
        }
