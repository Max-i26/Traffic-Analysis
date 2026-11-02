"""
Module B v2: Custom Multi-Scale FCOS Vehicle Detector.
Fully Convolutional One-Stage Object Detection across P2 (stride 4), P3 (stride 8), and P4 (stride 16).
Separate classification and regression towers with shared weights and level-specific scale parameters.
Strictly designed for real-time edge/laptop GPU execution (NVIDIA RTX 2050 4 GB VRAM).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

class Scale(nn.Module):
    """Learnable scale multiplier per feature pyramid level for FCOS regression."""
    def __init__(self, init_val: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class FeaturePyramidNetwork(nn.Module):
    """
    Constructs multi-scale representations P2, P3, P4 from backbone stages C2, C3, C4.
    """
    def __init__(self, in_channels_list: List[int], out_channels: int = 96):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        outputs = [fpn_conv(lat) for fpn_conv, lat in zip(self.fpn_convs, laterals)]
        return outputs  # [P2 (s=4), P3 (s=8), P4 (s=16)]

class CustomFCOSDetector(nn.Module):
    """
    Module B v2: Fully Convolutional One-Stage Vehicle Detector.
    Features:
    - Backbone: 4-stage convolutional downsampling (strides 2, 4, 8, 16)
    - Neck: Feature Pyramid Network generating P2, P3, P4 (96 channels each)
    - Shared Towers:
        * 2-conv Classification Tower with Dropout -> (B, num_classes, H, W)
        * 2-conv Regression Tower with Dropout -> (B, 4, H, W) [l, t, r, b]
        * Centerness Head -> (B, 1, H, W)
    - Stride normalization support: predicts dimensionless relative offsets.
    """
    def __init__(self,
                 num_classes: int = 4,
                 fpn_channels: int = 96,
                 tower_channels: int = 96,
                 dropout: float = 0.1,
                 normalize_by_stride: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_strides = [4, 8, 16]
        self.normalize_by_stride = normalize_by_stride

        # Backbone Stages
        self.stage1 = nn.Sequential(
            ConvNormAct(3, 32, stride=2),   # /2
            ConvNormAct(32, 32)
        )
        self.stage2 = nn.Sequential(
            ConvNormAct(32, 64, stride=2),  # /4 (C2)
            ConvNormAct(64, 64)
        )
        self.stage3 = nn.Sequential(
            ConvNormAct(64, 128, stride=2), # /8 (C3)
            ConvNormAct(128, 128)
        )
        self.stage4 = nn.Sequential(
            ConvNormAct(128, 256, stride=2),# /16 (C4)
            ConvNormAct(256, 256)
        )

        # FPN Neck (C2: 64, C3: 128, C4: 256) -> fpn_channels
        self.neck = FeaturePyramidNetwork([64, 128, 256], out_channels=fpn_channels)

        # Shared Towers with Dropout for regularization against overfitting
        self.cls_tower = nn.Sequential(
            ConvNormAct(fpn_channels, tower_channels),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            ConvNormAct(tower_channels, tower_channels)
        )
        self.reg_tower = nn.Sequential(
            ConvNormAct(fpn_channels, tower_channels),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            ConvNormAct(tower_channels, tower_channels)
        )

        # Level-shared output heads
        self.cls_head = nn.Conv2d(tower_channels, num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(tower_channels, 4, kernel_size=3, padding=1)
        self.cnt_head = nn.Conv2d(tower_channels, 1, kernel_size=3, padding=1)

        # Level-specific learnable scales for LTRB regression
        self.scales = nn.ModuleList([Scale(init_val=1.0) for _ in self.fpn_strides])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Focal loss prior initialization:
        # std=0.01 on weights ensures logits don't overpower the bias at step 0
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=0.01)
        prior_prob = 0.01
        bias_init = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_init)

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        features = self.neck([c2, c3, c4])

        all_cls_logits = []
        all_bbox_preds = []
        all_cnt_logits = []

        for i, feat in enumerate(features):
            stride = self.fpn_strides[i]

            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            cls_logits = self.cls_head(cls_feat)
            cnt_logits = self.cnt_head(reg_feat)

            # Regression: predict dimensionless LTRB (normalized by stride) or raw pixels
            reg_raw = self.reg_head(reg_feat)
            reg_scaled = self.scales[i](reg_raw)
            if self.normalize_by_stride:
                bbox_preds = F.relu(reg_scaled) # Dimensionless distance
            else:
                bbox_preds = F.relu(reg_scaled) * float(stride) # Raw pixels

            all_cls_logits.append(cls_logits)
            all_bbox_preds.append(bbox_preds)
            all_cnt_logits.append(cnt_logits)

        return {
            'cls_logits': all_cls_logits,
            'bbox_preds': all_bbox_preds,
            'cnt_logits': all_cnt_logits
        }
