"""
Module B: Custom Multi-Scale Vehicle Detector.
Custom CNN Backbone + Feature Pyramid Network (FPN) + Anchor-Free Dense Detection Head.
Detects 4 core traffic categories: Car, Bus, Truck, Motorcycle.
Optimized for NVIDIA RTX 2050 (4 GB VRAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

class FeaturePyramidNetwork(nn.Module):
    """
    Lightweight Feature Pyramid Network (FPN).
    Fuses deep semantic features with shallow high-resolution spatial features.
    """
    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        # features: [C2, C3, C4] from shallow to deep
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        # Smooth lateral features
        outputs = [fpn_conv(lat) for fpn_conv, lat in zip(self.fpn_convs, laterals)]
        return outputs # [P2, P3, P4]

class CustomVehicleDetector(nn.Module):
    """
    Custom Anchor-Free Single-Stage Vehicle Detector.
    Backbone: 4-stage convolutional downsampling.
    Neck: Feature Pyramid Network (FPN).
    Head: Shared dense regression and multi-class classification heads.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

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

        # FPN Neck (C2: 64, C3: 128, C4: 256) -> 128 channels
        self.neck = FeaturePyramidNetwork([64, 128, 256], out_channels=128)

        # Multi-scale Detection Head operating on P3 (/8 stride, primary vehicle scale)
        self.cls_head = nn.Sequential(
            ConvNormAct(128, 128),
            ConvNormAct(128, 64),
            nn.Conv2d(64, num_classes, kernel_size=1) # Logits per class
        )
        self.reg_head = nn.Sequential(
            ConvNormAct(128, 128),
            ConvNormAct(128, 64),
            nn.Conv2d(64, 4, kernel_size=1), # Bbox: [cx_norm, cy_norm, w_norm, h_norm]
            nn.Sigmoid()
        )
        self.obj_head = nn.Sequential(
            ConvNormAct(128, 64),
            nn.Conv2d(64, 1, kernel_size=1) # Objectness confidence
        )

    def forward(self, x: torch.Tensor):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        p2, p3, p4 = self.neck([c2, c3, c4])

        # Feature level P3 (stride 8: e.g. 36x64 grid for 288x512 input)
        cls_logits = self.cls_head(p3)
        bbox_preds = self.reg_head(p3)
        obj_logits = self.obj_head(p3)

        return {
            'cls_logits': cls_logits, # [B, num_classes, H/8, W/8]
            'bbox_preds': bbox_preds, # [B, 4, H/8, W/8]
            'obj_logits': obj_logits  # [B, 1, H/8, W/8]
        }
