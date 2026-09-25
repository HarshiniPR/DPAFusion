"""
Complementarity Estimation Module (CEM) (Stage I, Section 3.1.3)
Reference: DAPFusion Framework - Stage I: Multi-Modal Feature Representation Backbone

The Complementarity Estimation Module evaluates channel-wise cross-modal
complementarity between context-enriched RGB features (Fr) and Thermal features (Ft).
It computes a holistic descriptor combining absolute differences and multiplicative
interactions between pooled modality descriptors, producing a continuous channel
weight vector Sc in [0, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn


class ComplementarityEstimationModule(nn.Module):
    """
    Complementarity Estimation Module (CEM).
    Corresponds to Stage I, Section 3.1.3:
    - Step 1: Global Average Pool Fr and Ft -> zr, zt (B, 128)
    - Step 2: Dc = concat([zr, zt, |zr - zt|, zr * zt], dim=-1) -> (B, 512)
    - Step 3: Linear(512, 256) -> ReLU -> Linear(256, 128) -> Sigmoid -> Sc (B, 128)
    - Returns (Sc, zr, zt)
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.descriptor_dim = 4 * in_channels

        self.mlp = nn.Sequential(
            nn.Linear(self.descriptor_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self, Fr: torch.Tensor, Ft: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert Fr.ndim == 4 and Ft.ndim == 4
        assert Fr.shape == Ft.shape
        B, C, H, W = Fr.shape
        assert C == self.in_channels

        zr = Fr.mean(dim=[-2, -1])
        zt = Ft.mean(dim=[-2, -1])
        assert zr.shape == (B, self.in_channels)
        assert zt.shape == (B, self.in_channels)

        diff_term = torch.abs(zr - zt)
        prod_term = zr * zt
        Dc = torch.cat([zr, zt, diff_term, prod_term], dim=-1)
        assert Dc.shape == (B, self.descriptor_dim)

        Sc = self.mlp(Dc)
        assert Sc.shape == (B, self.in_channels)
        assert (Sc >= 0.0).all() and (Sc <= 1.0).all()

        return Sc, zr, zt


if __name__ == "__main__":
    print("Testing CEM...")
    cem = ComplementarityEstimationModule(128, 256)
    Fr = torch.randn(2, 128, 64, 64)
    Ft = torch.randn(2, 128, 64, 64)
    Sc, zr, zt = cem(Fr, Ft)
    assert Sc.shape == (2, 128)
    print("CEM verified successfully!")
