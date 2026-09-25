"""
Shallow CNN Feature Extraction Module (Stage I, Section 3.1.1)
Reference: DAPFusion Framework - Stage I: Multi-Modal Feature Representation Backbone

This module extracts low-level local spatial features and edge primitives from
individual imaging modalities (RGB or Thermal).

The stem performs spatial downsampling using strided convolutions:

    Input:  (B, C_in, H, W)
    Conv1:  (B, 32, H, W)
    Conv2:  (B, 64, H/2, W/2)
    Conv3:  (B, 128, H/4, W/4)

For the default 256x256 input:

    (B, C_in, 256, 256)
        -> (B, 32, 256, 256)
        -> (B, 64, 128, 128)
        -> (B, 128, 64, 64)

For the image-size ablation, the same architecture supports 128x128,
192x192, and 256x256 inputs without changing the network structure.

No pooling layers are used.
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

    - Conv1: 3x3, stride=1
    - Conv2: 3x3, stride=2
    - Conv3: 3x3, stride=2

    Each convolution is followed by:
        BatchNorm2d
        LeakyReLU(0.1)

    The architecture performs a total 4x spatial downsampling.

    Args:
        in_channels:
            Number of input channels.
            3 for RGB and 1 for Thermal.

        c1:
            Output channels of Conv1. Default = 32.

        c2:
            Output channels of Conv2. Default = 64.

        c3:
            Output channels of Conv3. Default = 128.

        negative_slope:
            Negative slope of LeakyReLU. Default = 0.1.
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

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if min(c1, c2, c3) <= 0:
            raise ValueError("Stem channel sizes must be positive.")

        self.in_channels = in_channels
        self.out_channels = c3

        # -------------------------------------------------------------
        # Conv 1
        # Spatial resolution is preserved.
        #
        # (B, in_channels, H, W)
        #       ->
        # (B, c1, H, W)
        # -------------------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = nn.LeakyReLU(
            negative_slope=negative_slope,
            inplace=True,
        )

        # -------------------------------------------------------------
        # Conv 2
        # 2x spatial downsampling.
        #
        # (B, c1, H, W)
        #       ->
        # (B, c2, H/2, W/2)
        # -------------------------------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.LeakyReLU(
            negative_slope=negative_slope,
            inplace=True,
        )

        # -------------------------------------------------------------
        # Conv 3
        # Additional 2x spatial downsampling.
        #
        # (B, c2, H/2, W/2)
        #       ->
        # (B, c3, H/4, W/4)
        # -------------------------------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels=c2,
            out_channels=c3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(c3)
        self.act3 = nn.LeakyReLU(
            negative_slope=negative_slope,
            inplace=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Supports arbitrary spatial dimensions as long as H and W are
        compatible with the two stride-2 convolutions.

        Examples:

            128x128 -> 32x32
            192x192 -> 48x48
            256x256 -> 64x64
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), "
                f"got shape {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected in_channels={self.in_channels}, "
                f"got {x.shape[1]}"
            )

        H, W = x.shape[-2:]

        if H < 4 or W < 4:
            raise ValueError(
                f"Input spatial dimensions must be >= 4, "
                f"got {H}x{W}"
            )

        # Conv 1
        h1 = self.act1(
            self.bn1(
                self.conv1(x)
            )
        )

        # Conv 2
        h2 = self.act2(
            self.bn2(
                self.conv2(h1)
            )
        )

        # Conv 3
        out = self.act3(
            self.bn3(
                self.conv3(h2)
            )
        )

        # The two stride-2 convolutions should reduce the spatial
        # dimensions approximately by a factor of four.
        expected_h = (H + 3) // 4
        expected_w = (W + 3) // 4

        if out.shape[-2:] != (expected_h, expected_w):
            raise RuntimeError(
                "Unexpected CNN stem spatial size: "
                f"input={H}x{W}, "
                f"output={out.shape[-2]}x{out.shape[-1]}, "
                f"expected={expected_h}x{expected_w}"
            )

        if out.shape[1] != self.out_channels:
            raise RuntimeError(
                f"Unexpected CNN stem channel count: "
                f"expected={self.out_channels}, "
                f"got={out.shape[1]}"
            )

        return out


if __name__ == "__main__":
    print("Testing CNNStem...")

    stem = CNNStem(in_channels=3)
    stem.eval()

    test_sizes = [128, 192, 256]

    with torch.no_grad():
        for size in test_sizes:
            dummy = torch.randn(1, 3, size, size)
            output = stem(dummy)

            expected = (size + 3) // 4

            print(
                f"Input:  {tuple(dummy.shape)}"
                f" -> Output: {tuple(output.shape)}"
            )

            assert output.shape == (
                1,
                128,
                expected,
                expected,
            )

    print("CNNStem verified successfully!")