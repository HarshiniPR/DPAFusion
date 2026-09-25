"""
Unified Stage-I DAPFusion Model
================================

Reference:
    DAPFusion Framework - Stage I:
    Multi-Modal Feature Representation Backbone

Stage-I pipeline:

    RGB
      │
      ▼
    CNN Stem
      │
      ▼
    Vision Mamba Context Encoder
      │
      ▼
    RGB Context Feature Fr
      │
      │
      ├───────────────┐
      │               │
      │               ▼
      │              CEM
      │               │
      │               ▼
      │              Sc
      │               │
      ▼               ▼
    Thermal       Cross-Modal
      │            Interaction
      ▼               │
    CNN Stem           │
      │               │
      ▼               │
    Vision Mamba       │
      │               │
      ▼               │
    Thermal Context   │
    Feature Ft ───────┘
                      │
                      ▼
              Fr' and Ft'
                      │
                      ▼
              Unified Fusion
                      │
                      ▼
                     Fu

The model is resolution-independent.

Supported image-size experiments:

    128 x 128
    192 x 192
    256 x 256

For the default three-layer CNN stem:

    Image size     Feature size entering Mamba
    ------------------------------------------------
    128 x 128      32 x 32
    192 x 192      48 x 48
    256 x 256      64 x 64

No fixed 256x256 assumption is used in the model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

_project_root = str(
    Path(__file__).resolve().parents[2]
)

if _project_root not in sys.path:
    sys.path.insert(
        0,
        _project_root,
    )

import torch
import torch.nn as nn

from dapfusion.stage1.cnn_stem import CNNStem
from dapfusion.stage1.complementarity import (
    ComplementarityEstimationModule,
)
from dapfusion.stage1.config import Stage1Config
from dapfusion.stage1.cross_interaction import (
    CrossModalInteraction,
)
from dapfusion.stage1.mamba_encoder import (
    VisionMambaEncoder,
)


# =====================================================================
# Unified Feature Fusion
# =====================================================================

class UnifiedFeatureFusion(nn.Module):
    """
    Unified Feature Representation Fusion Layer.

    Stage I, Section 3.1.5.

    The complementarity-enhanced RGB and Thermal features are
    concatenated:

        [Fr' || Ft']

    followed by a 1x1 convolution:

        Conv1x1(2C -> C)

    followed by:

        BatchNorm
        LeakyReLU

    Args:
        in_channels:
            Number of concatenated input channels.
            Default = 256.

        out_channels:
            Number of output unified feature channels.
            Default = 128.

        activation:
            Either:

                "bn_leaky_relu"
                "none"

        leaky_relu_slope:
            Negative slope for LeakyReLU.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 128,
        activation: str = "bn_leaky_relu",
        leaky_relu_slope: float = 0.1,
    ) -> None:

        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive."
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels must be positive."
            )

        if activation not in (
            "bn_leaky_relu",
            "none",
        ):
            raise ValueError(
                "activation must be either "
                "'bn_leaky_relu' or 'none'."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation

        # -------------------------------------------------------------
        # 1x1 Fusion Convolution
        # -------------------------------------------------------------
        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=(
                activation == "none"
            ),
        )

        # -------------------------------------------------------------
        # Normalization + activation
        # -------------------------------------------------------------
        if activation == "bn_leaky_relu":

            self.bn = nn.BatchNorm2d(
                out_channels
            )

            self.act = nn.LeakyReLU(
                negative_slope=leaky_relu_slope,
                inplace=True,
            )

        else:

            self.bn = nn.Identity()
            self.act = nn.Identity()

    def forward(
        self,
        Fr_prime: torch.Tensor,
        Ft_prime: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse complementarity-enhanced RGB and Thermal features.

        Args:
            Fr_prime:
                RGB enhanced feature map:

                    (B, C, H, W)

            Ft_prime:
                Thermal enhanced feature map:

                    (B, C, H, W)

        Returns:
            Fu:
                Unified feature representation:

                    (B, out_channels, H, W)
        """

        if Fr_prime.ndim != 4:
            raise ValueError(
                "Fr_prime must have shape "
                "(B, C, H, W), "
                f"got {tuple(Fr_prime.shape)}"
            )

        if Ft_prime.ndim != 4:
            raise ValueError(
                "Ft_prime must have shape "
                "(B, C, H, W), "
                f"got {tuple(Ft_prime.shape)}"
            )

        if Fr_prime.shape != Ft_prime.shape:
            raise ValueError(
                "RGB and Thermal enhanced features must "
                "have identical shapes. "
                f"Got RGB={tuple(Fr_prime.shape)}, "
                f"Thermal={tuple(Ft_prime.shape)}"
            )

        B, C, H, W = Fr_prime.shape

        # -------------------------------------------------------------
        # Concatenate modality features.
        #
        # (B,C,H,W) + (B,C,H,W)
        #          ->
        # (B,2C,H,W)
        # -------------------------------------------------------------
        concat_features = torch.cat(
            [
                Fr_prime,
                Ft_prime,
            ],
            dim=1,
        )

        if concat_features.shape[1] != (
            self.in_channels
        ):
            raise ValueError(
                "Unexpected number of fusion input channels. "
                f"Expected {self.in_channels}, "
                f"got {concat_features.shape[1]}"
            )

        # -------------------------------------------------------------
        # 1x1 convolution + optional BN + activation.
        # -------------------------------------------------------------
        Fu = self.act(
            self.bn(
                self.conv1x1(
                    concat_features
                )
            )
        )

        expected_shape = (
            B,
            self.out_channels,
            H,
            W,
        )

        if Fu.shape != expected_shape:
            raise RuntimeError(
                "Unexpected unified feature shape. "
                f"Expected {expected_shape}, "
                f"got {tuple(Fu.shape)}"
            )

        return Fu


# =====================================================================
# Stage-I Model
# =====================================================================

class Stage1Model(nn.Module):
    """
    DAPFusion Stage-I Feature Representation Backbone.

    This class wires together:

        1. RGB CNN Stem
        2. Thermal CNN Stem
        3. RGB Vision Mamba Encoder
        4. Thermal Vision Mamba Encoder
        5. Complementarity Estimation Module
        6. Complementarity-Guided Cross-Modal Interaction
        7. Unified Feature Fusion

    The output dictionary contains intermediate representations needed
    by the Stage-I losses and by subsequent Stage-II processing.
    """

    def __init__(
        self,
        config: Optional[Stage1Config] = None,
    ) -> None:

        super().__init__()

        self.config = (
            config
            if config is not None
            else Stage1Config()
        )

        # -------------------------------------------------------------
        # Read architecture dimensions from config.
        # -------------------------------------------------------------
        stem_c1 = (
            self.config.stem_channels[0]
        )

        stem_c2 = (
            self.config.stem_channels[1]
        )

        stem_c3 = (
            self.config.stem_channels[2]
        )

        # -------------------------------------------------------------
        # The Mamba encoder operates on the final CNN stem channel
        # dimension.
        #
        # The current methodology uses:
        #
        #     CNN -> 128 channels -> Mamba d_model=128
        #
        # Do not silently project between different dimensions here.
        # If the configuration is inconsistent, fail explicitly.
        # -------------------------------------------------------------
        if stem_c3 != self.config.d_model:
            raise ValueError(
                "Stage-I configuration mismatch: "
                f"stem_channels[-1]={stem_c3}, "
                f"but d_model={self.config.d_model}. "
                "The current architecture expects these to match."
            )

        # -------------------------------------------------------------
        # Unified fusion input.
        #
        # Fr' and Ft' each have d_model channels.
        #
        # Therefore:
        #
        #     fusion input = 2 * d_model
        # -------------------------------------------------------------
        expected_fusion_channels = (
            2 * self.config.d_model
        )

        if (
            self.config.fusion_in_channels
            != expected_fusion_channels
        ):
            raise ValueError(
                "Stage-I fusion configuration mismatch: "
                f"expected fusion_in_channels="
                f"{expected_fusion_channels}, "
                f"got "
                f"{self.config.fusion_in_channels}."
            )

        # =============================================================
        # 1. RGB CNN Stem
        # =============================================================

        self.rgb_stem = CNNStem(
            in_channels=(
                self.config.rgb_in_channels
            ),
            c1=stem_c1,
            c2=stem_c2,
            c3=stem_c3,
            negative_slope=(
                self.config.stem_leaky_relu_slope
            ),
        )

        # =============================================================
        # 2. Thermal CNN Stem
        # =============================================================

        self.thermal_stem = CNNStem(
            in_channels=(
                self.config.thermal_in_channels
            ),
            c1=stem_c1,
            c2=stem_c2,
            c3=stem_c3,
            negative_slope=(
                self.config.stem_leaky_relu_slope
            ),
        )

        # -------------------------------------------------------------
        # Mamba spatial size
        #
        # The CNN stem performs two stride-2 operations.
        #
        # For each spatial dimension:
        #
        #     output = ceil(input / 4)
        #
        # For the supported experiments:
        #
        #     128 -> 32
        #     192 -> 48
        #     256 -> 64
        #
        # No 256-specific assumption is used.
        # -------------------------------------------------------------
        mamba_height = (
            self.config.image_height + 3
        ) // 4

        mamba_width = (
            self.config.image_width + 3
        ) // 4

        self.mamba_spatial_size = (
            mamba_height,
            mamba_width,
        )

        # =============================================================
        # 3. RGB Vision Mamba Context Encoder
        # =============================================================

        self.rgb_encoder = VisionMambaEncoder(
            d_model=self.config.d_model,
            d_state=self.config.d_state,
            expand_ratio=self.config.expand_ratio,
            dropout=self.config.mamba_dropout,
            num_blocks=self.config.num_mamba_blocks,
            backend=self.config.mamba_backend,
            num_scan_directions=(
                self.config.num_scan_directions
            ),
            scan_merge=self.config.scan_merge,
            spatial_size=(
                self.mamba_spatial_size
            ),
        )

        # =============================================================
        # 4. Thermal Vision Mamba Context Encoder
        # =============================================================

        self.thermal_encoder = VisionMambaEncoder(
            d_model=self.config.d_model,
            d_state=self.config.d_state,
            expand_ratio=self.config.expand_ratio,
            dropout=self.config.mamba_dropout,
            num_blocks=self.config.num_mamba_blocks,
            backend=self.config.mamba_backend,
            num_scan_directions=(
                self.config.num_scan_directions
            ),
            scan_merge=self.config.scan_merge,
            spatial_size=(
                self.mamba_spatial_size
            ),
        )

        # =============================================================
        # 5. Complementarity Estimation Module
        # =============================================================

        self.cem = (
            ComplementarityEstimationModule(
                in_channels=(
                    self.config.d_model
                ),
                hidden_dim=(
                    self.config.cem_hidden_dim
                ),
            )
        )

        # =============================================================
        # 6. Complementarity-Guided Cross-Modal Interaction
        # =============================================================

        self.interaction = (
            CrossModalInteraction(
                init_alpha=(
                    self.config.init_alpha
                ),
                num_channels=(
                    self.config.d_model
                ),
            )
        )

        # =============================================================
        # 7. Unified Feature Fusion
        # =============================================================

        self.fusion = UnifiedFeatureFusion(
            in_channels=(
                self.config.fusion_in_channels
            ),
            out_channels=(
                self.config.fusion_out_channels
            ),
            activation=(
                self.config.fusion_activation
            ),
            leaky_relu_slope=(
                self.config.fusion_leaky_relu_slope
            ),
        )

    # =================================================================
    # Forward
    # =================================================================

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the complete Stage-I representation pipeline.

        Args:
            rgb:
                RGB input tensor:

                    (B, 3, H, W)

            thermal:
                Thermal input tensor:

                    (B, 1, H, W)

        Returns:
            Dictionary containing:

                fused
                Fr
                Ft
                Fr_prime
                Ft_prime
                zr
                zt
                Sc
                cnn_rgb
                cnn_thermal
        """

        # -------------------------------------------------------------
        # Input validation
        # -------------------------------------------------------------
        if not isinstance(
            rgb,
            torch.Tensor,
        ):
            raise TypeError(
                "rgb must be a torch.Tensor."
            )

        if not isinstance(
            thermal,
            torch.Tensor,
        ):
            raise TypeError(
                "thermal must be a torch.Tensor."
            )

        if rgb.ndim != 4:
            raise ValueError(
                "RGB input must have shape "
                "(B, C, H, W), "
                f"got {tuple(rgb.shape)}"
            )

        if thermal.ndim != 4:
            raise ValueError(
                "Thermal input must have shape "
                "(B, C, H, W), "
                f"got {tuple(thermal.shape)}"
            )

        # -------------------------------------------------------------
        # Channel validation
        # -------------------------------------------------------------
        if (
            rgb.shape[1]
            != self.config.rgb_in_channels
        ):
            raise ValueError(
                "Unexpected RGB channel count. "
                f"Expected "
                f"{self.config.rgb_in_channels}, "
                f"got {rgb.shape[1]}"
            )

        if (
            thermal.shape[1]
            != self.config.thermal_in_channels
        ):
            raise ValueError(
                "Unexpected Thermal channel count. "
                f"Expected "
                f"{self.config.thermal_in_channels}, "
                f"got {thermal.shape[1]}"
            )

        # -------------------------------------------------------------
        # Batch and spatial validation
        # -------------------------------------------------------------
        B, _, H, W = rgb.shape

        if thermal.shape[0] != B:
            raise ValueError(
                "RGB and Thermal batch sizes must match. "
                f"Got RGB={B}, "
                f"Thermal={thermal.shape[0]}"
            )

        if thermal.shape[-2:] != (
            H,
            W,
        ):
            raise ValueError(
                "RGB and Thermal spatial dimensions must match. "
                f"Got RGB={H}x{W}, "
                f"Thermal="
                f"{thermal.shape[-2]}x"
                f"{thermal.shape[-1]}"
            )

        # -------------------------------------------------------------
        # The model is configured for one image resolution at a time.
        #
        # This allows the same architecture to be instantiated for
        # 128, 192, and 256 image-size experiments.
        # -------------------------------------------------------------
        expected_input_size = (
            self.config.image_height,
            self.config.image_width,
        )

        if (
            H,
            W
        ) != expected_input_size:

            raise ValueError(
                "Input image size does not match the current "
                "Stage1Config. "
                f"Expected "
                f"{expected_input_size[0]}x"
                f"{expected_input_size[1]}, "
                f"got {H}x{W}."
            )

        # =============================================================
        # Stage 1.1 — CNN Feature Extraction
        # =============================================================

        cnn_rgb = self.rgb_stem(
            rgb
        )

        cnn_thermal = self.thermal_stem(
            thermal
        )

        # -------------------------------------------------------------
        # Determine expected CNN spatial dimensions.
        # -------------------------------------------------------------
        expected_feature_size = (
            self.mamba_spatial_size
        )

        expected_cnn_shape = (
            B,
            self.config.d_model,
            expected_feature_size[0],
            expected_feature_size[1],
        )

        if cnn_rgb.shape != expected_cnn_shape:
            raise RuntimeError(
                "Unexpected RGB CNN feature shape. "
                f"Expected {expected_cnn_shape}, "
                f"got {tuple(cnn_rgb.shape)}"
            )

        if cnn_thermal.shape != expected_cnn_shape:
            raise RuntimeError(
                "Unexpected Thermal CNN feature shape. "
                f"Expected {expected_cnn_shape}, "
                f"got {tuple(cnn_thermal.shape)}"
            )

        # =============================================================
        # Stage 1.2 — Dual Mamba Context Encoding
        # =============================================================

        Fr = self.rgb_encoder(
            cnn_rgb
        )

        Ft = self.thermal_encoder(
            cnn_thermal
        )

        if Fr.shape != expected_cnn_shape:
            raise RuntimeError(
                "Unexpected RGB Mamba feature shape. "
                f"Expected {expected_cnn_shape}, "
                f"got {tuple(Fr.shape)}"
            )

        if Ft.shape != expected_cnn_shape:
            raise RuntimeError(
                "Unexpected Thermal Mamba feature shape. "
                f"Expected {expected_cnn_shape}, "
                f"got {tuple(Ft.shape)}"
            )

        # =============================================================
        # Stage 1.3 — Complementarity Estimation
        # =============================================================

        Sc, zr, zt = self.cem(
            Fr,
            Ft,
        )

        expected_descriptor_shape = (
            B,
            self.config.d_model,
        )

        if Sc.shape != expected_descriptor_shape:
            raise RuntimeError(
                "Unexpected complementarity score shape. "
                f"Expected {expected_descriptor_shape}, "
                f"got {tuple(Sc.shape)}"
            )

        if zr.shape != expected_descriptor_shape:
            raise RuntimeError(
                "Unexpected RGB descriptor shape. "
                f"Expected {expected_descriptor_shape}, "
                f"got {tuple(zr.shape)}"
            )

        if zt.shape != expected_descriptor_shape:
            raise RuntimeError(
                "Unexpected Thermal descriptor shape. "
                f"Expected {expected_descriptor_shape}, "
                f"got {tuple(zt.shape)}"
            )

        # =============================================================
        # Stage 1.4 — Complementarity-Guided Interaction
        # =============================================================

        Fr_prime, Ft_prime = (
            self.interaction(
                Fr,
                Ft,
                Sc,
            )
        )

        if Fr_prime.shape != expected_cnn_shape:
            raise RuntimeError(
                "Unexpected enhanced RGB feature shape. "
                f"Expected {expected_cnn_shape}, "
                f"got {tuple(Fr_prime.shape)}"
            )

        if Ft_prime.shape != expected_cnn_shape:
            raise RuntimeError(
                "Unexpected enhanced Thermal feature shape. "
                f"Expected {expected_cnn_shape}, "
                f"got {tuple(Ft_prime.shape)}"
            )

        # =============================================================
        # Stage 1.5 — Unified Feature Representation
        # =============================================================

        Fu = self.fusion(
            Fr_prime,
            Ft_prime,
        )

        expected_fused_shape = (
            B,
            self.config.fusion_out_channels,
            expected_feature_size[0],
            expected_feature_size[1],
        )

        if Fu.shape != expected_fused_shape:
            raise RuntimeError(
                "Unexpected unified feature shape. "
                f"Expected {expected_fused_shape}, "
                f"got {tuple(Fu.shape)}"
            )

        # =============================================================
        # Return all representations required by Stage-I losses
        # and Stage-II processing.
        # =============================================================

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

    # =================================================================
    # Convenience interface
    # =================================================================

    def get_fused_features(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return only the unified Stage-I feature representation.

        This interface is intended for Stage-II RL policy integration.
        """

        return self.forward(
            rgb,
            thermal,
        )["fused"]


# =====================================================================
# Standalone verification
# =====================================================================

def _run_shape_test(
    image_size: int,
) -> None:
    """
    Test Stage1Model at one image resolution.

    Uses one Mamba block for a lightweight structural test.
    The actual configuration can use the full number of blocks.
    """

    config = Stage1Config(
        image_height=image_size,
        image_width=image_size,
        num_mamba_blocks=1,
    )

    model = Stage1Model(
        config=config
    )

    model.eval()

    rgb = torch.randn(
        1,
        3,
        image_size,
        image_size,
    )

    thermal = torch.randn(
        1,
        1,
        image_size,
        image_size,
    )

    with torch.no_grad():

        outputs = model(
            rgb,
            thermal,
        )

    feature_size = (
        image_size + 3
    ) // 4

    expected_feature_shape = (
        1,
        128,
        feature_size,
        feature_size,
    )

    expected_descriptor_shape = (
        1,
        128,
    )

    assert outputs[
        "cnn_rgb"
    ].shape == expected_feature_shape

    assert outputs[
        "cnn_thermal"
    ].shape == expected_feature_shape

    assert outputs[
        "Fr"
    ].shape == expected_feature_shape

    assert outputs[
        "Ft"
    ].shape == expected_feature_shape

    assert outputs[
        "Fr_prime"
    ].shape == expected_feature_shape

    assert outputs[
        "Ft_prime"
    ].shape == expected_feature_shape

    assert outputs[
        "fused"
    ].shape == expected_feature_shape

    assert outputs[
        "Sc"
    ].shape == expected_descriptor_shape

    assert outputs[
        "zr"
    ].shape == expected_descriptor_shape

    assert outputs[
        "zt"
    ].shape == expected_descriptor_shape

    # -------------------------------------------------------------
    # Verify numerical finiteness.
    # -------------------------------------------------------------
    for name, value in outputs.items():

        assert torch.isfinite(
            value
        ).all(), (
            f"Non-finite output in {name}"
        )

    print(
        f"Stage1Model {image_size}x{image_size}: "
        f"PASS | "
        f"feature={feature_size}x{feature_size}"
    )


if __name__ == "__main__":

    print(
        "Testing Stage1Model..."
    )

    # -------------------------------------------------------------
    # Test all intended image-size configurations.
    # -------------------------------------------------------------
    for size in (
        128,
        192,
        256,
    ):

        _run_shape_test(
            size
        )

    print(
        "Stage1Model verified successfully!"
    )