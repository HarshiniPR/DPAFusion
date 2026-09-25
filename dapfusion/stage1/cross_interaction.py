"""
Complementarity-Guided Cross-Modal Interaction Module (Stage I, Section 3.1.4)
Reference: DAPFusion Framework - Stage I: Multi-Modal Feature Representation Backbone

This module uses channel-wise complementarity weights (Sc) to selectively inject
salient complementary features across RGB (Fr) and Thermal (Ft) modalities through
a symmetric residual interaction controlled by a learnable scaling parameter alpha:
    Fr_prime = Fr + alpha * (Sc_broadcast * Ft)
    Ft_prime = Ft + alpha * (Sc_broadcast * Fr)
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


class CrossModalInteraction(nn.Module):
    """Complementarity-Guided Cross-Modal Interaction. Corresponds to Stage I, Section 3.1.4."""

    def __init__(
        self,
        init_alpha: float = 1.0,
        num_channels: int = 128,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(
        self, Fr: torch.Tensor, Ft: torch.Tensor, Sc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert Fr.ndim == 4 and Ft.ndim == 4
        assert Fr.shape == Ft.shape
        assert Sc.ndim == 2

        B, C, H, W = Fr.shape
        assert C == self.num_channels
        assert Sc.shape == (B, C)

        Sc_broadcast = Sc.view(B, C, 1, 1)

        Fr_prime = Fr + self.alpha * (Sc_broadcast * Ft)
        Ft_prime = Ft + self.alpha * (Sc_broadcast * Fr)

        assert Fr_prime.shape == (B, C, H, W)
        assert Ft_prime.shape == (B, C, H, W)

        return Fr_prime, Ft_prime


if __name__ == "__main__":
    print("Testing CrossModalInteraction...")
    cmi = CrossModalInteraction()
    Fr = torch.randn(2, 128, 64, 64)
    Ft = torch.randn(2, 128, 64, 64)
    Sc = torch.rand(2, 128)
    Fr_p, Ft_p = cmi(Fr, Ft, Sc)
    assert Fr_p.shape == (2, 128, 64, 64)
    print("CrossModalInteraction verified successfully!")
