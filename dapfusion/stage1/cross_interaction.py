"""
Complementarity-Guided Cross-Modal Interaction Module
=====================================================

Stage I, Section 3.1.4

Reference:
    DAPFusion Framework - Stage I:
    Multi-Modal Feature Representation Backbone

The module uses the channel-wise complementarity scores Sc estimated
by the Complementarity Estimation Module (CEM) to selectively inject
complementary information between RGB and Thermal feature maps.

Given:

    Fr : RGB feature map
    Ft : Thermal feature map
    Sc : channel-wise complementarity scores

the symmetric residual interaction is:

    Fr' = Fr + alpha * (Sc ⊙ Ft)

    Ft' = Ft + alpha * (Sc ⊙ Fr)

where:

    alpha

is a learnable scalar controlling the strength of cross-modal
information exchange.

The module is resolution-independent and therefore supports the
Stage-I image-size ablation:

    128 x 128
    192 x 192
    256 x 256

No spatial resolution is hard-coded.
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


class CrossModalInteraction(nn.Module):
    """
    Complementarity-Guided Cross-Modal Interaction.

    Stage I, Section 3.1.4.

    The module performs symmetric residual cross-modal interaction:

        Fr' = Fr + alpha * (Sc * Ft)

        Ft' = Ft + alpha * (Sc * Fr)

    where Sc is broadcast from:

        (B, C)

    to:

        (B, C, 1, 1)

    before being applied channel-wise to the feature maps.

    Args:
        init_alpha:
            Initial value of the learnable interaction coefficient.
            Default = 1.0.

        num_channels:
            Number of feature channels.
            Default = 128.
    """

    def __init__(
        self,
        init_alpha: float = 1.0,
        num_channels: int = 128,
    ) -> None:
        super().__init__()

        if num_channels <= 0:
            raise ValueError(
                f"num_channels must be positive, got {num_channels}"
            )

        if not torch.isfinite(
            torch.tensor(init_alpha, dtype=torch.float32)
        ):
            raise ValueError(
                f"init_alpha must be finite, got {init_alpha}"
            )

        self.num_channels = num_channels

        # -------------------------------------------------------------
        # Learnable cross-modal interaction strength.
        #
        # This remains a scalar parameter as specified by the
        # methodology, rather than changing it into a channel-wise
        # or spatial attention mechanism.
        # -------------------------------------------------------------
        self.alpha = nn.Parameter(
            torch.tensor(
                init_alpha,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        Fr: torch.Tensor,
        Ft: torch.Tensor,
        Sc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform complementarity-guided cross-modal interaction.

        Args:
            Fr:
                RGB feature map.

                Shape:
                    (B, C, H, W)

            Ft:
                Thermal feature map.

                Shape:
                    (B, C, H, W)

            Sc:
                Channel-wise complementarity scores.

                Shape:
                    (B, C)

                Values are expected to be in [0, 1], as produced by
                the sigmoid output of the CEM.

        Returns:
            Fr_prime:
                Complementarity-enhanced RGB features.

                Shape:
                    (B, C, H, W)

            Ft_prime:
                Complementarity-enhanced Thermal features.

                Shape:
                    (B, C, H, W)
        """

        # -------------------------------------------------------------
        # Input validation
        # -------------------------------------------------------------
        if Fr.ndim != 4:
            raise ValueError(
                "Fr must have shape (B, C, H, W), "
                f"got {tuple(Fr.shape)}"
            )

        if Ft.ndim != 4:
            raise ValueError(
                "Ft must have shape (B, C, H, W), "
                f"got {tuple(Ft.shape)}"
            )

        if Sc.ndim != 2:
            raise ValueError(
                "Sc must have shape (B, C), "
                f"got {tuple(Sc.shape)}"
            )

        if Fr.shape != Ft.shape:
            raise ValueError(
                "RGB and Thermal feature maps must have identical "
                f"shapes. Got RGB={tuple(Fr.shape)}, "
                f"Thermal={tuple(Ft.shape)}"
            )

        B, C, H, W = Fr.shape

        if C != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} feature channels, "
                f"got {C}"
            )

        if Sc.shape != (B, C):
            raise ValueError(
                "Complementarity score shape must be "
                f"(B, C)=({B}, {C}), "
                f"got {tuple(Sc.shape)}"
            )

        # -------------------------------------------------------------
        # Broadcast channel-wise complementarity scores.
        #
        # Sc:
        #     (B, C)
        #
        # becomes:
        #     (B, C, 1, 1)
        #
        # so every spatial location of a channel receives the same
        # learned complementarity weight.
        # -------------------------------------------------------------
        Sc_broadcast = Sc.reshape(
            B,
            C,
            1,
            1,
        )

        # -------------------------------------------------------------
        # Symmetric residual cross-modal interaction.
        #
        # Fr' = Fr + alpha * (Sc * Ft)
        #
        # Ft' = Ft + alpha * (Sc * Fr)
        # -------------------------------------------------------------
        Fr_prime = (
            Fr
            + self.alpha
            * (
                Sc_broadcast
                * Ft
            )
        )

        Ft_prime = (
            Ft
            + self.alpha
            * (
                Sc_broadcast
                * Fr
            )
        )

        return Fr_prime, Ft_prime


# =====================================================================
# Standalone verification
# =====================================================================

if __name__ == "__main__":

    print("Testing CrossModalInteraction...")

    torch.manual_seed(42)

    cmi = CrossModalInteraction(
        init_alpha=1.0,
        num_channels=128,
    )

    cmi.eval()

    # -------------------------------------------------------------
    # The module is spatial-resolution independent.
    #
    # CEM output corresponding to:
    #
    #   128x128 image -> 32x32 feature map
    #   192x192 image -> 48x48 feature map
    #   256x256 image -> 64x64 feature map
    #
    # -------------------------------------------------------------
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

            Sc = torch.rand(
                2,
                128,
            )

            Fr_prime, Ft_prime = cmi(
                Fr,
                Ft,
                Sc,
            )

            expected_shape = (
                2,
                128,
                feature_size,
                feature_size,
            )

            assert Fr_prime.shape == expected_shape
            assert Ft_prime.shape == expected_shape

            assert torch.isfinite(
                Fr_prime
            ).all()

            assert torch.isfinite(
                Ft_prime
            ).all()

            print(
                f"Feature size {feature_size}x{feature_size}: "
                f"Fr'={tuple(Fr_prime.shape)}, "
                f"Ft'={tuple(Ft_prime.shape)}"
            )

    # -------------------------------------------------------------
    # Verify alpha is learnable.
    # -------------------------------------------------------------
    assert isinstance(
        cmi.alpha,
        nn.Parameter,
    )

    assert cmi.alpha.requires_grad

    print(
        f"Learnable alpha: "
        f"{cmi.alpha.detach().item():.4f}"
    )

    # -------------------------------------------------------------
    # Verify the residual interaction mathematically.
    # -------------------------------------------------------------
    Fr = torch.ones(
        1,
        128,
        4,
        4,
    )

    Ft = torch.ones(
        1,
        128,
        4,
        4,
    )

    Sc = torch.ones(
        1,
        128,
    )

    cmi_test = CrossModalInteraction(
        init_alpha=1.0,
        num_channels=128,
    )

    cmi_test.eval()

    with torch.no_grad():

        Fr_prime, Ft_prime = cmi_test(
            Fr,
            Ft,
            Sc,
        )

    # 1 + 1 * (1 * 1) = 2
    assert torch.allclose(
        Fr_prime,
        torch.full_like(Fr_prime, 2.0),
    )

    assert torch.allclose(
        Ft_prime,
        torch.full_like(Ft_prime, 2.0),
    )

    print(
        "Symmetric residual interaction verified."
    )

    print(
        "CrossModalInteraction verified successfully!"
    )