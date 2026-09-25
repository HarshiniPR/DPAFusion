"""
Stage 1 Loss Functions Module (Stage I, Section 3.1.6)
Reference: DAPFusion Framework - Stage I: Multi-Modal Feature Representation Backbone
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from dapfusion.stage1.config import Stage1Config


def sobel_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute per-channel Sobel spatial gradient magnitude: |grad_x| + |grad_y|."""
    assert x.ndim == 4, f"Expected 4D tensor (B, C, H, W), got {tuple(x.shape)}"
    B, C, H, W = x.shape

    sobel_x_kernel = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        dtype=x.dtype,
        device=x.device,
    )
    sobel_y_kernel = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]],
        dtype=x.dtype,
        device=x.device,
    )

    weight_x = sobel_x_kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    weight_y = sobel_y_kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    x_padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    grad_x = F.conv2d(x_padded, weight_x, groups=C)
    grad_y = F.conv2d(x_padded, weight_y, groups=C)

    mag = torch.abs(grad_x) + torch.abs(grad_y)
    assert mag.shape == (B, C, H, W)
    return mag


def normalize_features(
    x: torch.Tensor,
    norm_type: Literal["l2", "instance"] = "l2",
) -> torch.Tensor:
    """Apply feature normalization before loss calculation."""
    if norm_type == "l2":
        return F.normalize(x, p=2, dim=1, eps=1e-6)
    elif norm_type == "instance":
        return F.instance_norm(x, eps=1e-5)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


def structure_preservation_loss(
    Fu: torch.Tensor, Fr: torch.Tensor, Ft: torch.Tensor
) -> torch.Tensor:
    """Structure Preservation Loss (Stage I, Section 3.1.6a)."""
    grad_fu = sobel_gradient_magnitude(Fu)
    grad_fr = sobel_gradient_magnitude(Fr)
    grad_ft = sobel_gradient_magnitude(Ft)

    target = torch.maximum(grad_fr, grad_ft)
    return F.l1_loss(grad_fu, target, reduction="mean")


def feature_decorrelation_loss(
    zr: torch.Tensor, zt: torch.Tensor, tau: float = 0.3
) -> torch.Tensor:
    """Feature Decorrelation Loss (Stage I, Section 3.1.6b)."""
    assert zr.shape == zt.shape
    cos_sim = F.cosine_similarity(zr, zt, dim=-1)
    loss_per_sample = torch.clamp(cos_sim - tau, min=0.0)
    return loss_per_sample.mean()


def complementarity_diversity_loss(
    Sc: torch.Tensor,
    target_mean: float = 0.5,
    min_std: float = 0.05,
) -> torch.Tensor:
    """Complementarity Diversity Loss (Stage I, Section 3.1.6c)."""
    assert Sc.ndim == 2
    sc_mean = Sc.mean(dim=-1)
    sc_std = Sc.std(dim=-1, unbiased=False) + 1e-7

    mean_penalty = torch.abs(sc_mean - target_mean)
    std_penalty = torch.clamp(min_std - sc_std, min=0.0)
    return (mean_penalty + std_penalty).mean()


def feature_information_preservation_loss(
    Fu: torch.Tensor, ref: torch.Tensor
) -> torch.Tensor:
    """Feature Information Preservation Loss (Stage I, Section 3.1.6d)."""
    assert Fu.shape == ref.shape
    return F.l1_loss(Fu, ref, reduction="mean")


class Stage1Loss(nn.Module):
    """Composite Multi-Objective Loss Module for DAPFusion Stage 1."""

    def __init__(self, config: Optional[Stage1Config] = None) -> None:
        super().__init__()
        self.config = config or Stage1Config()

        self.lambda_structure = self.config.lambda_structure
        self.lambda_decorr = self.config.lambda_decorr
        self.lambda_comp = self.config.lambda_comp
        self.lambda_info = self.config.lambda_info

        self.decorr_tau = self.config.decorr_tau
        self.comp_target_mean = self.config.comp_target_mean
        self.comp_min_std = self.config.comp_min_std

        self.info_loss_reference = self.config.info_loss_reference
        self.normalize_before_loss = self.config.normalize_before_loss
        self.feature_norm_type = self.config.feature_norm_type

    def forward(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        Fu = model_outputs["fused"]
        Fr = model_outputs["Fr"]
        Ft = model_outputs["Ft"]
        zr = model_outputs["zr"]
        zt = model_outputs["zt"]
        Sc = model_outputs["Sc"]

        if self.normalize_before_loss:
            Fu_norm = normalize_features(Fu, norm_type=self.feature_norm_type)
            Fr_norm = normalize_features(Fr, norm_type=self.feature_norm_type)
            Ft_norm = normalize_features(Ft, norm_type=self.feature_norm_type)
        else:
            Fu_norm, Fr_norm, Ft_norm = Fu, Fr, Ft

        loss_structure = structure_preservation_loss(Fu_norm, Fr_norm, Ft_norm)
        loss_decorr = feature_decorrelation_loss(zr, zt, tau=self.decorr_tau)
        loss_comp = complementarity_diversity_loss(
            Sc, target_mean=self.comp_target_mean, min_std=self.comp_min_std
        )

        if self.info_loss_reference == "original_cnn_output":
            cnn_rgb = model_outputs["cnn_rgb"]
            cnn_thermal = model_outputs["cnn_thermal"]
            if self.normalize_before_loss:
                ref_rgb = normalize_features(cnn_rgb, norm_type=self.feature_norm_type)
                ref_thermal = normalize_features(cnn_thermal, norm_type=self.feature_norm_type)
            else:
                ref_rgb, ref_thermal = cnn_rgb, cnn_thermal
            ref = (ref_rgb + ref_thermal) / 2.0
        else:
            ref = (Fr_norm + Ft_norm) / 2.0

        loss_info = feature_information_preservation_loss(Fu_norm, ref)

        loss_total = (
            self.lambda_structure * loss_structure
            + self.lambda_decorr * loss_decorr
            + self.lambda_comp * loss_comp
            + self.lambda_info * loss_info
        )

        return {
            "loss_total": loss_total,
            "loss_structure": loss_structure,
            "loss_decorr": loss_decorr,
            "loss_comp": loss_comp,
            "loss_info": loss_info,
        }


if __name__ == "__main__":
    print("Testing losses...")
    criterion = Stage1Loss()
    outputs = {
        "fused": torch.randn(2, 128, 64, 64),
        "Fr": torch.randn(2, 128, 64, 64),
        "Ft": torch.randn(2, 128, 64, 64),
        "zr": torch.randn(2, 128),
        "zt": torch.randn(2, 128),
        "Sc": torch.rand(2, 128),
        "cnn_rgb": torch.randn(2, 128, 64, 64),
        "cnn_thermal": torch.randn(2, 128, 64, 64),
    }
    l = criterion(outputs)
    assert "loss_total" in l
    print("Losses verified successfully!")
