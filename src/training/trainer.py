"""
PyTorch Training and Evaluation Engine.
Built with Automatic Mixed Precision (AMP) for RTX 2050 4GB GPU optimization.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from src.training.losses import BCEDiceLoss
from src.evaluation.metrics import compute_batch_metrics

class SegmentationTrainer:
    """
    Modular Trainer for Lane Segmentation Networks.
    Supports mixed precision (AMP), gradient scaling, and physical VRAM monitoring.
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: str = 'checkpoints',
                 model_name: str = 'model'):

        self.device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = criterion or BCEDiceLoss(bce_weight=0.5, dice_weight=1.0)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))

        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_iou = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for images, masks, _ in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Automatic Mixed Precision forward pass
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            # Scaled backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / max(1, len(self.train_loader))

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        metrics_sum = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'pixel_acc': 0.0}

        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                total_loss += loss.item()
                batch_m = compute_batch_metrics(logits, masks)
                for k in metrics_sum:
                    metrics_sum[k] += batch_m[k]

        num_batches = max(1, len(self.val_loader))
        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
        avg_metrics['loss'] = total_loss / num_batches
        return avg_metrics

    def fit(self, epochs: int = 5) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Training [{self.model_name}] on {self.device} for {epochs} Epochs")
        print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.evaluate()

            epoch_time = time.time() - epoch_start
            vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2) if self.device.type == 'cuda' else 0.0

            val_iou = val_metrics['iou']
            val_dice = val_metrics['dice']
            val_loss = val_metrics['loss']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_iou)
            self.history['val_dice'].append(val_dice)

            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
                best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_iou': val_iou,
                    'val_dice': val_dice
                }, best_path)

            best_indicator = " [*BEST]" if is_best else ""
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val IoU: {val_iou:.4f} | "
                  f"Val Dice: {val_dice:.4f} | "
                  f"VRAM: {vram_mb:.1f}MB | "
                  f"Time: {epoch_time:.2f}s{best_indicator}")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s. Best Val IoU: {self.best_val_iou:.4f}")
        return self.history
