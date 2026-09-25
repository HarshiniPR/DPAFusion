"""
Complementarity Estimation Module (CEM)
=======================================

Stage I, Section 3.1.3 of DAPFusion.

The Complementarity Estimation Module estimates channel-wise
cross-modal complementarity between RGB and Thermal features.

Given:
    Fr : RGB feature map
    Ft : Thermal feature map

the module computes global modality descriptors:

    zr = GAP(Fr)
    zt = GAP(Ft)

and constructs the complementarity descriptor:

    Dc = [zr, zt, |zr - zt|, zr ⊙ zt]

The descriptor is passed through an MLP:

    Linear(4C -> hidden_dim)
    ReLU
    Linear(hidden_dim -> C)
    Sigmoid

to produce:

    Sc ∈ [0, 1]^C

Sc represents the learned channel-wise complementarity between
the RGB and Thermal modalities.

The implementation supports arbitrary spatial resolutions, so
the same module works with the Stage-I image-size ablation:

    128x128 -> feature maps of corresponding spatial size
    192x192 -> feature maps of corresponding spatial size
    256x256 -> feature maps of corresponding spatial size

No spatial resolution is hard-coded here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------
# Project-root import handling
# ---------------------------------------------------------------------
_project_root = str(Path(__file__).resolve().parents[2])

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn


class ComplementarityEstimationModule(nn.Module):
    """
    Complementarity Estimation Module (CEM).

    Methodology:

        Fr, Ft
          │
          ├── Global Average Pooling
          │
          ├── zr, zt
          │
          └── Dc = [zr, zt, |zr-zt|, zr⊙zt]
                         │
                         ▼
                  Linear(4C -> hidden)
                         │
                       ReLU
                         │
                  Linear(hidden -> C)
                         │
                      Sigmoid
                         │
                         ▼
                        Sc

    Args:
        in_channels:
            Number of feature channels.
            Default = 128.

        hidden_dim:
            Hidden dimension of the complementarity MLP.
            Default = 256.

    Returns:
        Sc:
            Channel-wise complementarity scores.
            Shape = (B, C)

        zr:
            Global RGB descriptor.
            Shape = (B, C)

        zt:
            Global Thermal descriptor.
            Shape = (B, C)
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Four descriptors are concatenated:
        #
        #   zr
        #   zt
        #   |zr - zt|
        #   zr * zt
        #
        # Therefore the descriptor dimension is 4C.
        self.descriptor_dim = 4 * in_channels

        # -----------------------------------------------------------------
        # Complementarity MLP
        # -----------------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(
                self.descriptor_dim,
                hidden_dim,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                in_channels,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        Fr: torch.Tensor,
        Ft: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate channel-wise RGB-Thermal complementarity.

        Args:
            Fr:
                RGB feature map of shape:

                    (B, C, H, W)

            Ft:
                Thermal feature map of shape:

                    (B, C, H, W)

        Returns:
            Sc:
                Complementarity scores:

                    (B, C)

                with values in [0, 1].

            zr:
                Global RGB descriptor:

                    (B, C)

            zt:
                Global Thermal descriptor:

                    (B, C)
        """

        # -------------------------------------------------------------
        # Input validation
        # -------------------------------------------------------------
        if Fr.ndim != 4:
            raise ValueError(
                "RGB feature tensor must have shape (B, C, H, W), "
                f"got {tuple(Fr.shape)}"
            )

        if Ft.ndim != 4:
            raise ValueError(
                "Thermal feature tensor must have shape (B, C, H, W), "
                f"got {tuple(Ft.shape)}"
            )

        if Fr.shape != Ft.shape:
            raise ValueError(
                "RGB and Thermal feature maps must have identical shapes. "
                f"Got RGB={tuple(Fr.shape)}, "
                f"Thermal={tuple(Ft.shape)}"
            )

        B, C, H, W = Fr.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} feature channels, "
                f"got {C}"
            )

        # -------------------------------------------------------------
        # Step 1: Global Average Pooling
        #
        # zr = GAP(Fr)
        # zt = GAP(Ft)
        #
        # Shape:
        #
        # (B, C, H, W) -> (B, C)
        # -------------------------------------------------------------
        zr = Fr.mean(dim=(-2, -1))
        zt = Ft.mean(dim=(-2, -1))

        # -------------------------------------------------------------
        # Step 2: Complementarity descriptor
        #
        # Dc =
        # [
        #     zr,
        #     zt,
        #     |zr - zt|,
        #     zr ⊙ zt
        # ]
        #
        # Shape:
        #
        # (B, C) × 4 -> (B, 4C)
        # -------------------------------------------------------------
        diff_term = torch.abs(zr - zt)

        prod_term = zr * zt

        Dc = torch.cat(
            [
                zr,
                zt,
                diff_term,
                prod_term,
            ],
            dim=-1,
        )

        # -------------------------------------------------------------
        # Step 3: Complementarity MLP
        #
        # (B, 4C)
        #      ->
        # (B, hidden_dim)
        #      ->
        # (B, C)
        #      ->
        # sigmoid
        #      ->
        # Sc ∈ [0,1]
        # -------------------------------------------------------------
        Sc = self.mlp(Dc)

        return Sc, zr, zt


# =====================================================================
# Standalone verification
# =====================================================================

if __name__ == "__main__":
    print("Testing ComplementarityEstimationModule...")

    torch.manual_seed(42)

    model = ComplementarityEstimationModule(
        in_channels=128,
        hidden_dim=256,
    )

    model.eval()

    # Test the three image-size configurations.
    #
    # The CEM itself is resolution-independent because it uses
    # global average pooling.
    test_feature_sizes = [32, 48, 64]

    with torch.no_grad():

        for feature_size in test_feature_sizes:

            Fr = torch.randn(
                2,
                128,
                feature_size,
                feature_size,
            )

            Ft = torch.randn(
                2,
                128,
                feature_size,
                feature_size,
            )

            Sc, zr, zt = model(Fr, Ft)

            assert Sc.shape == (2, 128)
            assert zr.shape == (2, 128)
            assert zt.shape == (2, 128)

            assert torch.isfinite(Sc).all()
            assert torch.isfinite(zr).all()
            assert torch.isfinite(zt).all()

            assert torch.all(Sc >= 0.0)
            assert torch.all(Sc <= 1.0)

            print(
                f"Feature size {feature_size}x{feature_size}: "
                f"Fr={tuple(Fr.shape)}, "
                f"Ft={tuple(Ft.shape)}, "
                f"Sc={tuple(Sc.shape)}"
            )

    print("CEM verified successfully!")