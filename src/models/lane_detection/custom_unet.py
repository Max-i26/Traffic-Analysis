"""
Model A2: Custom U-Net for Lane Detection.
Features multi-scale skip connections to preserve thin lane boundaries and markings.
Optimized for RTX 2050 4 GB VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> LeakyReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UpBlock(nn.Module):
    """Upsampling followed by concatenation with encoder skip connection and DoubleConv"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)

        # Handle any slight dimension mismatch due to odd input sizes
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x_cat = torch.cat([skip, x_up], dim=1)
        return self.conv(x_cat)

class CustomUNet(nn.Module):
    """
    Custom U-Net architecture for lane boundary segmentation.
    Includes 4 encoder stages, 1 bottleneck stage, and 4 decoder stages with skip connections.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()

        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2) # /2

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2) # /4

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2) # /8

        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2) # /16

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder (Upsampling with skip connections)
        self.up4 = UpBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.up3 = UpBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.up2 = UpBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.up1 = UpBlock(in_channels=64, skip_channels=32, out_channels=32)

        # Segmentation Head
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder passes
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        s4 = self.enc4(p3)
        p4 = self.pool4(s4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder passes with skip connections
        d4 = self.up4(b, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        logits = self.out_conv(d1)
        return logits
