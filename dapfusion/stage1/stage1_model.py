"""
Unified Stage 1 Model (Stage I, Section 3.1.5)
Reference: DAPFusion Framework - Stage I: Multi-Modal Feature Representation Backbone

Wires together all Stage 1 components:
1. Shallow CNN Feature Extraction for RGB and Thermal (cnn_stem.py)
2. Dual Vision Mamba Context Encoders (mamba_encoder.py)
3. Complementarity Estimation Module (complementarity.py)
4. Complementarity-Guided Cross-Modal Interaction (cross_interaction.py)
5. Unified Feature Representation Fusion Head (stage1_model.py)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn

from dapfusion.stage1.cnn_stem import CNNStem
from dapfusion.stage1.complementarity import ComplementarityEstimationModule
from dapfusion.stage1.config import Stage1Config
from dapfusion.stage1.cross_interaction import CrossModalInteraction
from dapfusion.stage1.mamba_encoder import VisionMambaEncoder


class UnifiedFeatureFusion(nn.Module):
    """Unified Feature Representation Fusion Layer. Corresponds to Stage I, Section 3.1.5."""

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 128,
        activation: str = "bn_leaky_relu",
        leaky_relu_slope: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation

        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=(activation == "none"),
        )

        if activation == "bn_leaky_relu":
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)
        else:
            self.bn = nn.Identity()
            self.act = nn.Identity()

    def forward(self, Fr_prime: torch.Tensor, Ft_prime: torch.Tensor) -> torch.Tensor:
        assert Fr_prime.shape == Ft_prime.shape
        B, C, H, W = Fr_prime.shape

        concat_features = torch.cat([Fr_prime, Ft_prime], dim=1)
        assert concat_features.shape == (B, self.in_channels, H, W)

        Fu = self.act(self.bn(self.conv1x1(concat_features)))
        assert Fu.shape == (B, self.out_channels, H, W)
        return Fu


class Stage1Model(nn.Module):
    """Stage 1 Feature Representation Backbone for DAPFusion."""

    def __init__(self, config: Optional[Stage1Config] = None) -> None:
        super().__init__()
        self.config = config or Stage1Config()

        self.rgb_stem = CNNStem(
            in_channels=self.config.rgb_in_channels,
            c1=self.config.stem_channels[0],
            c2=self.config.stem_channels[1],
            c3=self.config.stem_channels[2],
            negative_slope=self.config.stem_leaky_relu_slope,
        )
        self.thermal_stem = CNNStem(
            in_channels=self.config.thermal_in_channels,
            c1=self.config.stem_channels[0],
            c2=self.config.stem_channels[1],
            c3=self.config.stem_channels[2],
            negative_slope=self.config.stem_leaky_relu_slope,
        )

        self.rgb_encoder = VisionMambaEncoder(
            d_model=self.config.d_model,
            d_state=self.config.d_state,
            expand_ratio=self.config.expand_ratio,
            dropout=self.config.mamba_dropout,
            num_blocks=self.config.num_mamba_blocks,
            backend=self.config.mamba_backend,
            num_scan_directions=self.config.num_scan_directions,
            scan_merge=self.config.scan_merge,
            spatial_size=(64, 64),
        )
        self.thermal_encoder = VisionMambaEncoder(
            d_model=self.config.d_model,
            d_state=self.config.d_state,
            expand_ratio=self.config.expand_ratio,
            dropout=self.config.mamba_dropout,
            num_blocks=self.config.num_mamba_blocks,
            backend=self.config.mamba_backend,
            num_scan_directions=self.config.num_scan_directions,
            scan_merge=self.config.scan_merge,
            spatial_size=(64, 64),
        )

        self.cem = ComplementarityEstimationModule(
            in_channels=self.config.d_model,
            hidden_dim=self.config.cem_hidden_dim,
        )

        self.interaction = CrossModalInteraction(
            init_alpha=self.config.init_alpha,
            num_channels=self.config.d_model,
        )

        self.fusion = UnifiedFeatureFusion(
            in_channels=self.config.fusion_in_channels,
            out_channels=self.config.fusion_out_channels,
            activation=self.config.fusion_activation,
            leaky_relu_slope=self.config.fusion_leaky_relu_slope,
        )

    def forward(
        self, rgb: torch.Tensor, thermal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        assert rgb.ndim == 4 and thermal.ndim == 4
        assert rgb.shape[1] == self.config.rgb_in_channels
        assert thermal.shape[1] == self.config.thermal_in_channels
        B, _, H, W = rgb.shape
        assert thermal.shape[0] == B
        assert H == 256 and W == 256

        # 1. CNN stems
        cnn_rgb = self.rgb_stem(rgb)
        cnn_thermal = self.thermal_stem(thermal)
        assert cnn_rgb.shape == (B, 128, 64, 64)
        assert cnn_thermal.shape == (B, 128, 64, 64)

        # 2. Mamba context encoders
        Fr = self.rgb_encoder(cnn_rgb)
        Ft = self.thermal_encoder(cnn_thermal)
        assert Fr.shape == (B, 128, 64, 64)
        assert Ft.shape == (B, 128, 64, 64)

        # 3. CEM
        Sc, zr, zt = self.cem(Fr, Ft)
        assert Sc.shape == (B, 128)
        assert zr.shape == (B, 128)
        assert zt.shape == (B, 128)

        # 4. Cross interaction
        Fr_prime, Ft_prime = self.interaction(Fr, Ft, Sc)
        assert Fr_prime.shape == (B, 128, 64, 64)
        assert Ft_prime.shape == (B, 128, 64, 64)

        # 5. Fusion
        Fu = self.fusion(Fr_prime, Ft_prime)
        assert Fu.shape == (B, 128, 64, 64)

        return {
            "fused": Fu,
            "Fr": Fr,
            "Ft": Ft,
            "Fr_prime": Fr_prime,
            "Ft_prime": Ft_prime,
            "zr": zr,
            "zt": zt,
            "Sc": Sc,
            "cnn_rgb": cnn_rgb,
            "cnn_thermal": cnn_thermal,
        }

    def get_fused_features(
        self, rgb: torch.Tensor, thermal: torch.Tensor
    ) -> torch.Tensor:
        """Convenience interface directly returning Fu for Stage 2 RL policy."""
        return self.forward(rgb, thermal)["fused"]


if __name__ == "__main__":
    print("Testing Stage1Model...")
    cfg = Stage1Config(num_mamba_blocks=1)
    model = Stage1Model(config=cfg)
    rgb = torch.randn(2, 3, 256, 256)
    thermal = torch.randn(2, 1, 256, 256)
    out = model(rgb, thermal)
    assert out["fused"].shape == (2, 128, 64, 64)
    print("Stage1Model verified successfully!")
