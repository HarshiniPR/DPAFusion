"""
Shallow CNN Feature Extraction Module (Stage I, Section 3.1.1)
Reference: DAPFusion Framework - Stage I: Multi-Modal Feature Representation Backbone

This module extracts low-level local spatial features and edge primitives from
individual imaging modalities (RGB or Thermal). It performs initial spatial
downsampling from (B, C_in, 256, 256) to (B, 128, 64, 64) using strided convolutions
without pooling layers, preserving dense spatial representations.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn


class CNNStem(nn.Module):
    """
    Shallow CNN Stem for Modality-Specific Feature Extraction.

    Corresponds to Stage I, Section 3.1.1:
    - Conv1: 3x3, in_channels -> 32, stride=1, padding=1, BatchNorm2d, LeakyReLU(0.1)
    - Conv2: 3x3, 32 -> 64, stride=2, padding=1, BatchNorm2d, LeakyReLU(0.1)
    - Conv3: 3x3, 64 -> 128, stride=2, padding=1, BatchNorm2d, LeakyReLU(0.1)

    Downsampling is performed entirely via strided convolutions (stride=2 in Conv2 & Conv3).
    No pooling layers are utilized.

    Args:
        in_channels (int): Number of input channels (3 for RGB, 1 for Thermal).
        c1 (int): Output channels of first conv layer (default: 32).
        c2 (int): Output channels of second conv layer (default: 64).
        c3 (int): Output channels of third conv layer (default: 128).
        negative_slope (float): LeakyReLU negative slope (default: 0.1).
    """

    def __init__(
        self,
        in_channels: int,
        c1: int = 32,
        c2: int = 64,
        c3: int = 128,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = c3

        # Conv 1: (B, in_channels, 256, 256) -> (B, 32, 256, 256)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        # Conv 2: (B, 32, 256, 256) -> (B, 64, 128, 128)
        self.conv2 = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        # Conv 3: (B, 64, 128, 128) -> (B, 128, 64, 64)
        self.conv3 = nn.Conv2d(
            in_channels=c2,
            out_channels=c3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(c3)
        self.act3 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the shallow CNN stem."""
        assert x.ndim == 4, f"Expected 4D input tensor (B, C, H, W), got shape {tuple(x.shape)}"
        assert x.shape[1] == self.in_channels, (
            f"Expected in_channels={self.in_channels}, got {x.shape[1]}"
        )
        assert x.shape[2] == 256 and x.shape[3] == 256, (
            f"Expected spatial resolution 256x256, got {x.shape[2]}x{x.shape[3]}"
        )

        h1 = self.act1(self.bn1(self.conv1(x)))
        assert h1.shape[1:] == (32, 256, 256), f"Conv1 unexpected shape {tuple(h1.shape)}"

        h2 = self.act2(self.bn2(self.conv2(h1)))
        assert h2.shape[1:] == (64, 128, 128), f"Conv2 unexpected shape {tuple(h2.shape)}"

        out = self.act3(self.bn3(self.conv3(h2)))
        assert out.shape[1:] == (128, 64, 64), f"Conv3 output unexpected shape {tuple(out.shape)}"

        return out


if __name__ == "__main__":
    print("Testing CNNStem (Stage I, Section 3.1.1)...")
    batch_size = 2
    rgb_stem = CNNStem(in_channels=3)
    dummy_rgb = torch.randn(batch_size, 3, 256, 256)
    out_rgb = rgb_stem(dummy_rgb)
    assert out_rgb.shape == (batch_size, 128, 64, 64)
    print("CNNStem verified successfully!")
