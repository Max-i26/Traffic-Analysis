"""
Model A1: Custom Lightweight Convolutional Lane Segmentation Network.
Designed from scratch for the RTX 2050 (4 GB VRAM).
Sequential encoder-decoder architecture without skip connections (baseline comparison).
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class CustomCNNSegmentor(nn.Module):
    """
    Lightweight CNN lane segmentor (Model A1).
    Encoder downsamples by a factor of 16 (4 downsampling stages).
    Decoder progressively restores spatial resolution to [B, 1, H, W].
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2) # /2

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2) # /4

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2) # /8

        self.enc4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2) # /16

        # Bottleneck with dilated receptive field
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Decoder (Transposed convolutions without skip connections)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16, 16)

        # 1x1 classification head
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.pool1(self.enc1(x))
        e2 = self.pool2(self.enc2(e1))
        e3 = self.pool3(self.enc3(e2))
        e4 = self.pool4(self.enc4(e3))

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d4 = self.dec4(self.up4(b))
        d3 = self.dec3(self.up3(d4))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))

        logits = self.head(d1)
        return logits
