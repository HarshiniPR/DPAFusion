"""
Stage I Loss Functions
======================

DAPFusion Stage I, Section 3.1.6

The Stage-I objective contains four complementary losses:

    1. Structure Preservation Loss
    2. Feature Decorrelation Loss
    3. Complementarity Diversity Loss
    4. Feature Information Preservation Loss

Overall objective:

    L_total =
        lambda_structure * L_structure
        + lambda_decorr * L_decorr
        + lambda_comp * L_comp
        + lambda_info * L_info

The implementation is designed to remain numerically stable when
training with mixed precision (AMP).

Important:
    The methodology is unchanged. Numerical safeguards are used
    only where necessary to prevent NaN/Inf propagation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Literal, Optional

_project_root = str(Path(__file__).resolve().parents[2])

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from dapfusion.stage1.config import Stage1Config


# =====================================================================
# Utility validation
# =====================================================================

def _check_4d(
    x: torch.Tensor,
    name: str,
) -> None:
    """
    Validate a feature map has shape (B, C, H, W).
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if x.ndim != 4:
        raise ValueError(
            f"{name} must have shape (B, C, H, W), "
            f"got {tuple(x.shape)}"
        )


def _check_same_shape(
    x: torch.Tensor,
    y: torch.Tensor,
    x_name: str,
    y_name: str,
) -> None:
    """
    Validate that two tensors have identical shapes.
    """

    if x.shape != y.shape:
        raise ValueError(
            f"{x_name} and {y_name} must have identical shapes. "
            f"Got {tuple(x.shape)} and {tuple(y.shape)}."
        )


# =====================================================================
# Sobel gradient magnitude
# =====================================================================

def sobel_gradient_magnitude(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-channel Sobel spatial gradient magnitude.

    The structure descriptor is:

        |grad_x| + |grad_y|

    Args:
        x:
            Feature tensor of shape:

                (B, C, H, W)

    Returns:
        Gradient magnitude tensor with the same shape as x.
    """

    _check_4d(x, "x")

    B, C, H, W = x.shape

    if H < 2 or W < 2:
        raise ValueError(
            "Spatial dimensions must be at least 2x2 for "
            "Sobel gradient calculation."
        )

    # -------------------------------------------------------------
    # Numerical stability:
    #
    # Sobel kernels themselves are small integer-valued filters.
    # Under AMP, however, explicitly performing this operation in
    # float32 avoids unnecessary FP16 accumulation error.
    #
    # Gradients still propagate back to x through the cast.
    # -------------------------------------------------------------
    original_dtype = x.dtype

    x_fp32 = x.float()

    sobel_x_kernel = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=x.device,
    )

    sobel_y_kernel = torch.tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
        device=x.device,
    )

    weight_x = sobel_x_kernel.view(
        1,
        1,
        3,
        3,
    ).expand(
        C,
        1,
        3,
        3,
    )

    weight_y = sobel_y_kernel.view(
        1,
        1,
        3,
        3,
    ).expand(
        C,
        1,
        3,
        3,
    )

    x_padded = F.pad(
        x_fp32,
        (1, 1, 1, 1),
        mode="reflect",
    )

    grad_x = F.conv2d(
        x_padded,
        weight_x,
        groups=C,
    )

    grad_y = F.conv2d(
        x_padded,
        weight_y,
        groups=C,
    )

    magnitude = (
        torch.abs(grad_x)
        + torch.abs(grad_y)
    )

    if not torch.isfinite(magnitude).all():
        raise FloatingPointError(
            "Non-finite values encountered during "
            "Sobel gradient calculation."
        )

    # Return in the original dtype where possible. The loss itself
    # will still perform its final reduction in float32.
    return magnitude.to(original_dtype)


# =====================================================================
# Feature normalization
# =====================================================================

def normalize_features(
    x: torch.Tensor,
    norm_type: Literal[
        "l2",
        "instance",
    ] = "l2",
) -> torch.Tensor:
    """
    Normalize feature maps before loss calculation.

    Args:
        x:
            Feature tensor.

        norm_type:
            "l2"       -> channel-wise L2 normalization.
            "instance" -> InstanceNorm.

    Returns:
        Normalized tensor.
    """

    _check_4d(x, "x")

    if norm_type == "l2":

        # F.normalize is already numerically stabilized by eps.
        return F.normalize(
            x,
            p=2,
            dim=1,
            eps=1e-6,
        )

    if norm_type == "instance":

        # Instance normalization is performed in float32 to avoid
        # unstable reductions under FP16.
        normalized = F.instance_norm(
            x.float(),
            eps=1e-5,
        )

        return normalized.to(x.dtype)

    raise ValueError(
        f"Unknown norm_type: {norm_type}"
    )


# =====================================================================
# 1. Structure Preservation Loss
# =====================================================================

def structure_preservation_loss(
    Fu: torch.Tensor,
    Fr: torch.Tensor,
    Ft: torch.Tensor,
) -> torch.Tensor:
    """
    Structure Preservation Loss.

    Stage I, Section 3.1.6(a).

    The fused feature structure is encouraged to preserve the
    strongest spatial gradient information from either modality.

    Target:

        G_target = max(G_RGB, G_Thermal)

    Loss:

        L_structure =
            mean(
                |G_fused - G_target|
            )

    Args:
        Fu:
            Unified/fused feature map.

        Fr:
            RGB feature map.

        Ft:
            Thermal feature map.

    Returns:
        Scalar loss tensor.
    """

    _check_4d(Fu, "Fu")
    _check_4d(Fr, "Fr")
    _check_4d(Ft, "Ft")

    _check_same_shape(
        Fu,
        Fr,
        "Fu",
        "Fr",
    )

    _check_same_shape(
        Fu,
        Ft,
        "Fu",
        "Ft",
    )

    # -------------------------------------------------------------
    # Calculate spatial gradients.
    # -------------------------------------------------------------
    grad_fu = sobel_gradient_magnitude(Fu)
    grad_fr = sobel_gradient_magnitude(Fr)
    grad_ft = sobel_gradient_magnitude(Ft)

    # -------------------------------------------------------------
    # Strongest structural response from either modality.
    # -------------------------------------------------------------
    target = torch.maximum(
        grad_fr,
        grad_ft,
    )

    # -------------------------------------------------------------
    # Perform final reduction in float32.
    #
    # This avoids FP16 reduction overflow/underflow for larger
    # feature maps.
    # -------------------------------------------------------------
    loss = F.l1_loss(
        grad_fu.float(),
        target.float(),
        reduction="mean",
    )

    return loss


# =====================================================================
# 2. Feature Decorrelation Loss
# =====================================================================

def feature_decorrelation_loss(
    zr: torch.Tensor,
    zt: torch.Tensor,
    tau: float = 0.3,
) -> torch.Tensor:
    """
    Feature Decorrelation Loss.

    Stage I, Section 3.1.6(b).

    The loss penalizes excessive cosine similarity between RGB and
    Thermal global descriptors.

    Cosine similarity:

        cos(zr, zt)

    Penalty:

        max(cos(zr, zt) - tau, 0)

    Args:
        zr:
            RGB global descriptor of shape (B, C).

        zt:
            Thermal global descriptor of shape (B, C).

        tau:
            Similarity threshold.

    Returns:
        Scalar loss.
    """

    if not isinstance(zr, torch.Tensor):
        raise TypeError(
            "zr must be a torch.Tensor."
        )

    if not isinstance(zt, torch.Tensor):
        raise TypeError(
            "zt must be a torch.Tensor."
        )

    if zr.ndim != 2 or zt.ndim != 2:
        raise ValueError(
            "zr and zt must have shape (B, C). "
            f"Got zr={tuple(zr.shape)}, "
            f"zt={tuple(zt.shape)}."
        )

    _check_same_shape(
        zr,
        zt,
        "zr",
        "zt",
    )

    if not torch.isfinite(
        torch.tensor(
            tau,
            dtype=torch.float32,
        )
    ):
        raise ValueError(
            f"tau must be finite, got {tau}"
        )

    # -------------------------------------------------------------
    # Cosine similarity.
    #
    # Perform the operation in FP32 because it is a normalization
    # followed by a reduction and is therefore more sensitive to
    # low-precision numerical errors.
    # -------------------------------------------------------------
    zr_fp32 = zr.float()
    zt_fp32 = zt.float()

    cos_sim = F.cosine_similarity(
        zr_fp32,
        zt_fp32,
        dim=-1,
        eps=1e-8,
    )

    loss_per_sample = torch.clamp(
        cos_sim - float(tau),
        min=0.0,
    )

    return loss_per_sample.mean()


# =====================================================================
# 3. Complementarity Diversity Loss
# =====================================================================

def complementarity_diversity_loss(
    Sc: torch.Tensor,
    target_mean: float = 0.5,
    min_std: float = 0.05,
) -> torch.Tensor:
    """
    Complementarity Diversity Loss.

    Stage I, Section 3.1.6(c).

    The loss encourages the complementarity scores to:

        1. remain centered around target_mean
        2. retain a minimum amount of channel diversity

    Mean penalty:

        |mean(Sc) - target_mean|

    Standard-deviation penalty:

        max(min_std - std(Sc), 0)

    Args:
        Sc:
            Complementarity scores of shape (B, C).

        target_mean:
            Desired mean complementarity score.

        min_std:
            Minimum desired channel-wise variation.

    Returns:
        Scalar loss.
    """

    if not isinstance(Sc, torch.Tensor):
        raise TypeError(
            "Sc must be a torch.Tensor."
        )

    if Sc.ndim != 2:
        raise ValueError(
            "Sc must have shape (B, C), "
            f"got {tuple(Sc.shape)}"
        )

    # -------------------------------------------------------------
    # Use float32 for statistical reductions.
    # -------------------------------------------------------------
    Sc_fp32 = Sc.float()

    # -------------------------------------------------------------
    # Mean across channels.
    # -------------------------------------------------------------
    sc_mean = Sc_fp32.mean(
        dim=-1
    )

    # -------------------------------------------------------------
    # Population standard deviation.
    #
    # Instead of:
    #
    #     Sc.std(...)
    #
    # we explicitly compute:
    #
    #     sqrt(var + eps)
    #
    # This gives us direct control over the numerical stabilizer.
    # -------------------------------------------------------------
    sc_var = Sc_fp32.var(
        dim=-1,
        unbiased=False,
    )

    sc_std = torch.sqrt(
        sc_var + 1e-7
    )

    # -------------------------------------------------------------
    # Mean-centering penalty.
    # -------------------------------------------------------------
    mean_penalty = torch.abs(
        sc_mean - float(target_mean)
    )

    # -------------------------------------------------------------
    # Diversity penalty.
    # -------------------------------------------------------------
    std_penalty = torch.clamp(
        float(min_std) - sc_std,
        min=0.0,
    )

    loss = (
        mean_penalty
        + std_penalty
    ).mean()

    return loss


# =====================================================================
# 4. Feature Information Preservation Loss
# =====================================================================

def feature_information_preservation_loss(
    Fu: torch.Tensor,
    ref: torch.Tensor,
) -> torch.Tensor:
    """
    Feature Information Preservation Loss.

    Stage I, Section 3.1.6(d).

    The unified representation is encouraged to preserve information
    from the reference representation.

    Loss:

        L_info = mean(|Fu - ref|)

    Args:
        Fu:
            Unified/fused feature representation.

        ref:
            Reference feature representation.

    Returns:
        Scalar loss.
    """

    _check_4d(Fu, "Fu")
    _check_4d(ref, "ref")

    _check_same_shape(
        Fu,
        ref,
        "Fu",
        "ref",
    )

    return F.l1_loss(
        Fu.float(),
        ref.float(),
        reduction="mean",
    )


# =====================================================================
# Composite Stage-I Loss
# =====================================================================

class Stage1Loss(nn.Module):
    """
    Composite Multi-Objective Loss Module for DAPFusion Stage I.

    The module computes:

        L_structure
        L_decorr
        L_comp
        L_info

    and combines them using the configured loss weights.
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
        # Loss weights
        # -------------------------------------------------------------
        self.lambda_structure = (
            self.config.lambda_structure
        )

        self.lambda_decorr = (
            self.config.lambda_decorr
        )

        self.lambda_comp = (
            self.config.lambda_comp
        )

        self.lambda_info = (
            self.config.lambda_info
        )

        # -------------------------------------------------------------
        # Loss hyperparameters
        # -------------------------------------------------------------
        self.decorr_tau = (
            self.config.decorr_tau
        )

        self.comp_target_mean = (
            self.config.comp_target_mean
        )

        self.comp_min_std = (
            self.config.comp_min_std
        )

        # -------------------------------------------------------------
        # Information preservation reference
        # -------------------------------------------------------------
        self.info_loss_reference = (
            self.config.info_loss_reference
        )

        # -------------------------------------------------------------
        # Feature normalization
        # -------------------------------------------------------------
        self.normalize_before_loss = (
            self.config.normalize_before_loss
        )

        self.feature_norm_type = (
            self.config.feature_norm_type
        )

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all Stage-I losses.

        Required model outputs:

            fused
            Fr
            Ft
            zr
            zt
            Sc

        Additional outputs required when:

            info_loss_reference == "original_cnn_output"

        are:

            cnn_rgb
            cnn_thermal

        Returns:
            Dictionary containing:

                loss_total
                loss_structure
                loss_decorr
                loss_comp
                loss_info
        """

        required_keys = {
            "fused",
            "Fr",
            "Ft",
            "zr",
            "zt",
            "Sc",
        }

        missing = sorted(
            required_keys
            - set(model_outputs.keys())
        )

        if missing:
            raise KeyError(
                "Missing required model outputs: "
                + ", ".join(missing)
            )

        # -------------------------------------------------------------
        # Extract model outputs.
        # -------------------------------------------------------------
        Fu = model_outputs["fused"]
        Fr = model_outputs["Fr"]
        Ft = model_outputs["Ft"]
        zr = model_outputs["zr"]
        zt = model_outputs["zt"]
        Sc = model_outputs["Sc"]

        # -------------------------------------------------------------
        # Feature normalization.
        #
        # Keep this behavior from the original implementation.
        # -------------------------------------------------------------
        if self.normalize_before_loss:

            Fu_norm = normalize_features(
                Fu,
                norm_type=self.feature_norm_type,
            )

            Fr_norm = normalize_features(
                Fr,
                norm_type=self.feature_norm_type,
            )

            Ft_norm = normalize_features(
                Ft,
                norm_type=self.feature_norm_type,
            )

        else:

            Fu_norm = Fu
            Fr_norm = Fr
            Ft_norm = Ft

        # -------------------------------------------------------------
        # 1. Structure Preservation
        # -------------------------------------------------------------
        loss_structure = structure_preservation_loss(
            Fu_norm,
            Fr_norm,
            Ft_norm,
        )

        # -------------------------------------------------------------
        # 2. Feature Decorrelation
        # -------------------------------------------------------------
        loss_decorr = feature_decorrelation_loss(
            zr,
            zt,
            tau=self.decorr_tau,
        )

        # -------------------------------------------------------------
        # 3. Complementarity Diversity
        # -------------------------------------------------------------
        loss_comp = complementarity_diversity_loss(
            Sc,
            target_mean=self.comp_target_mean,
            min_std=self.comp_min_std,
        )

        # -------------------------------------------------------------
        # 4. Information Preservation
        # -------------------------------------------------------------
        if (
            self.info_loss_reference
            == "original_cnn_output"
        ):

            if (
                "cnn_rgb"
                not in model_outputs
            ):
                raise KeyError(
                    "info_loss_reference='original_cnn_output' "
                    "requires model_outputs['cnn_rgb']."
                )

            if (
                "cnn_thermal"
                not in model_outputs
            ):
                raise KeyError(
                    "info_loss_reference='original_cnn_output' "
                    "requires model_outputs['cnn_thermal']."
                )

            cnn_rgb = model_outputs[
                "cnn_rgb"
            ]

            cnn_thermal = model_outputs[
                "cnn_thermal"
            ]

            if self.normalize_before_loss:

                ref_rgb = normalize_features(
                    cnn_rgb,
                    norm_type=self.feature_norm_type,
                )

                ref_thermal = normalize_features(
                    cnn_thermal,
                    norm_type=self.feature_norm_type,
                )

            else:

                ref_rgb = cnn_rgb
                ref_thermal = cnn_thermal

            ref = (
                ref_rgb
                + ref_thermal
            ) / 2.0

        else:

            # ---------------------------------------------------------
            # Default reference:
            #
            #   ref = (Fr + Ft) / 2
            # ---------------------------------------------------------
            ref = (
                Fr_norm
                + Ft_norm
            ) / 2.0

        loss_info = (
            feature_information_preservation_loss(
                Fu_norm,
                ref,
            )
        )

        # -------------------------------------------------------------
        # Total Stage-I objective.
        #
        # IMPORTANT:
        # The four objectives and their configured weights are
        # unchanged.
        # -------------------------------------------------------------
        loss_total = (
            self.lambda_structure
            * loss_structure
            + self.lambda_decorr
            * loss_decorr
            + self.lambda_comp
            * loss_comp
            + self.lambda_info
            * loss_info
        )

        # -------------------------------------------------------------
        # Final numerical safety check.
        #
        # Do not silently replace NaN/Inf with zeros.
        # If a non-finite loss occurs, raise an explicit error so
        # training cannot continue with corrupted gradients.
        # -------------------------------------------------------------
        losses = {
            "loss_total": loss_total,
            "loss_structure": loss_structure,
            "loss_decorr": loss_decorr,
            "loss_comp": loss_comp,
            "loss_info": loss_info,
        }

        for name, value in losses.items():

            if not torch.isfinite(value):
                raise FloatingPointError(
                    f"Non-finite value detected in {name}: "
                    f"{value.detach().item()}"
                )

        return losses


# =====================================================================
# Standalone verification
# =====================================================================

if __name__ == "__main__":

    print("Testing Stage-I losses...")

    torch.manual_seed(42)

    criterion = Stage1Loss()

    # -------------------------------------------------------------
    # 128x128 experiment
    #
    # After the CNN stem:
    #
    #   128x128 -> 32x32
    #
    # -------------------------------------------------------------
    outputs_128 = {
        "fused": torch.randn(
            2,
            128,
            32,
            32,
        ),

        "Fr": torch.randn(
            2,
            128,
            32,
            32,
        ),

        "Ft": torch.randn(
            2,
            128,
            32,
            32,
        ),

        "zr": torch.randn(
            2,
            128,
        ),

        "zt": torch.randn(
            2,
            128,
        ),

        "Sc": torch.rand(
            2,
            128,
        ),

        "cnn_rgb": torch.randn(
            2,
            128,
            32,
            32,
        ),

        "cnn_thermal": torch.randn(
            2,
            128,
            32,
            32,
        ),
    }

    losses = criterion(
        outputs_128
    )

    required_loss_keys = {
        "loss_total",
        "loss_structure",
        "loss_decorr",
        "loss_comp",
        "loss_info",
    }

    assert required_loss_keys.issubset(
        losses.keys()
    )

    for name, value in losses.items():

        assert value.ndim == 0
        assert torch.isfinite(value)

        print(
            f"{name}: "
            f"{value.item():.6f}"
        )

    # -------------------------------------------------------------
    # Check gradient propagation.
    # -------------------------------------------------------------
    outputs_grad = {
        key: value.clone().requires_grad_()
        for key, value in outputs_128.items()
    }

    grad_losses = criterion(
        outputs_grad
    )

    grad_losses[
        "loss_total"
    ].backward()

    assert outputs_grad[
        "fused"
    ].grad is not None

    assert torch.isfinite(
        outputs_grad["fused"].grad
    ).all()

    print(
        "Gradient propagation verified."
    )

    # -------------------------------------------------------------
    # Test complementarity loss with constant scores.
    #
    # This verifies the sqrt(var + eps) implementation remains
    # finite even when the variance is exactly zero.
    # -------------------------------------------------------------
    constant_sc = torch.full(
        (2, 128),
        0.5,
        requires_grad=True,
    )

    comp_loss = complementarity_diversity_loss(
        constant_sc
    )

    assert torch.isfinite(
        comp_loss
    )

    comp_loss.backward()

    assert constant_sc.grad is not None
    assert torch.isfinite(
        constant_sc.grad
    ).all()

    print(
        "Complementarity numerical stability verified."
    )

    print(
        "Stage-I losses verified successfully!"
    )