"""
Loss functions for lane and road segmentation.
Addresses severe class imbalance (sparse lane pixels vs overwhelming background).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation:
        Dice = (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)
        Loss = 1 - Dice
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions [B, 1, H, W] -> [B, -1]
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        total = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice.mean()

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    BCE provides smooth pixel-wise gradients, while Dice directly maximizes mask overlap.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
