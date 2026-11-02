"""
Module C: Custom Drivable Road Area Segmentation Model.
Built with Atrous Spatial Pyramid Pooling (ASPP) to capture broad road surfaces.
Optimized for the NVIDIA RTX 2050 (4 GB VRAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)

class RoadSegmentor(nn.Module):
    """
    Lightweight Drivable Road Segmentor.
    Backbone: 4-stage convolutional encoder.
    Bridge: ASPP module to capture multi-scale road contexts (near vs far horizon).
    Decoder: Skip-connected upsampling to full resolution [B, 1, H, W].
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        # Stage 1: /2
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Stage 2: /4
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Stage 3: /8
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Stage 4: /16
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Multi-scale road context
        self.aspp = ASPP(in_channels=256, out_channels=128, atrous_rates=(3, 6, 9))

        # Decoder stages
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(96 + 64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Final classification head
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]

        # Encoder
        f1 = self.enc1(x)  # /2
        f2 = self.enc2(f1) # /4
        f3 = self.enc3(f2) # /8
        f4 = self.enc4(f3) # /16

        # Multi-scale bridge
        b = self.aspp(f4)

        # Decoder with skips
        u3 = F.interpolate(b, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, f3], dim=1))

        u2 = F.interpolate(d3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, f2], dim=1))

        u1 = F.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, f1], dim=1))

        out = F.interpolate(self.head(d1), size=orig_size, mode='bilinear', align_corners=False)
        return out
