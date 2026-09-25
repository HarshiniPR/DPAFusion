"""
Dual Mamba Context Encoder Module (Stage I, Section 3.1.2)
Reference: DAPFusion Framework - Stage I: Multi-Modal Feature Representation Backbone

This module models long-range spatial dependencies across the 2D feature map
using a 4-directional selective state-space scan (VMamba-style SS2D). It features:
- Pre-LayerNorm architecture
- Linear feature expansion (expansion ratio = 2)
- Input-dependent Selective State-Space Scan (S6)
- Configurable scan directions (1 or 4: raster, reverse-raster, column-major, reverse-column-major)
- Dual backends:
    * "mamba_ssm": Uses official CUDA-accelerated Mamba package if available.
    * "custom_ssm": Pure PyTorch implementation with exact analytical gradient recurrence.
- 4 stacked blocks per modality encoder with residual connections.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MAMBA_SSM_AVAILABLE = False
try:
    from mamba_ssm import Mamba as MambaOfficial  # type: ignore
    MAMBA_SSM_AVAILABLE = True
except (ImportError, Exception):
    MAMBA_SSM_AVAILABLE = False


class FastSelectiveScanRecurrence(torch.autograd.Function):
    """Custom Autograd Function for Selective State-Space Recurrence."""

    @staticmethod
    def forward(
        ctx: Any,
        dA: torch.Tensor,
        dB_x: torch.Tensor,
        C_mat: torch.Tensor,
        D_param: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D, N = dA.shape
        h = torch.zeros(B, D, N, device=dA.device, dtype=dA.dtype)
        h_all = torch.empty(B, L, D, N, device=dA.device, dtype=dA.dtype)
        y = torch.empty(B, L, D, device=dA.device, dtype=dA.dtype)

        for t in range(L):
            h = dA[:, t] * h + dB_x[:, t]
            h_all[:, t] = h
            y[:, t] = (h * C_mat[:, t].unsqueeze(1)).sum(dim=-1) + x[:, t] * D_param

        ctx.save_for_backward(dA, dB_x, C_mat, D_param, x, h_all)
        return y

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        dA, dB_x, C_mat, D_param, x, h_all = ctx.saved_tensors
        B, L, D, N = dA.shape

        grad_dA = torch.empty_like(dA)
        grad_dB_x = torch.empty_like(dB_x)
        grad_C_mat = torch.empty_like(C_mat)
        grad_D_param = torch.zeros_like(D_param)
        grad_x = torch.empty_like(x)

        grad_h = torch.zeros(B, D, N, device=dA.device, dtype=dA.dtype)
        for t in range(L - 1, -1, -1):
            gy_t = grad_y[:, t]
            grad_D_param += (gy_t * x[:, t]).sum(dim=0)
            grad_x[:, t] = gy_t * D_param
            grad_C_mat[:, t] = (gy_t.unsqueeze(-1) * h_all[:, t]).sum(dim=1)

            grad_h = grad_h + gy_t.unsqueeze(-1) * C_mat[:, t].unsqueeze(1)
            grad_dB_x[:, t] = grad_h
            h_prev = (
                h_all[:, t - 1]
                if t > 0
                else torch.zeros(B, D, N, device=dA.device, dtype=dA.dtype)
            )
            grad_dA[:, t] = grad_h * h_prev
            grad_h = grad_h * dA[:, t]

        return grad_dA, grad_dB_x, grad_C_mat, grad_D_param, grad_x


class CustomSelectiveScanSSM(nn.Module):
    """Pure PyTorch fallback implementation of Selective State-Space Scan (S6)."""

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A_init = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(
            self.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        assert u.ndim == 3 and u.shape[-1] == self.d_model, (
            f"Expected shape (B, L, {self.d_model}), got {tuple(u.shape)}"
        )
        B, L, _ = u.shape

        xz = self.in_proj(u)
        x_branch, z_gate = torch.split(xz, [self.d_inner, self.d_inner], dim=-1)

        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L]
        x_conv = F.silu(x_conv).transpose(1, 2)

        x_proj_out = self.x_proj(x_conv)
        dt_raw, B_mat, C_mat = torch.split(
            x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))

        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)
        dB_x = dB * x_conv.unsqueeze(-1)

        y = FastSelectiveScanRecurrence.apply(dA, dB_x, C_mat, self.D, x_conv)
        y_gated = y * F.silu(z_gate)
        out = self.out_proj(y_gated)
        assert out.shape == (B, L, self.d_model), f"Shape mismatch: {tuple(out.shape)}"
        return out


class VisionMambaBlock(nn.Module):
    """Vision Mamba Block with VMamba-Style Multi-Directional Selective Scan."""

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        expand_ratio: int = 2,
        dropout: float = 0.1,
        backend: Literal["mamba_ssm", "custom_ssm"] = "custom_ssm",
        num_scan_directions: int = 4,
        scan_merge: Literal["average", "concat"] = "average",
        spatial_size: Tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand_ratio = expand_ratio
        self.num_scan_directions = num_scan_directions
        self.scan_merge = scan_merge
        self.spatial_size = spatial_size
        self.active_backend = backend

        assert num_scan_directions in (1, 4)
        assert scan_merge in ("average", "concat")

        self.norm = nn.LayerNorm(d_model)

        if backend == "mamba_ssm" and MAMBA_SSM_AVAILABLE:
            self.ssm = MambaOfficial(
                d_model=d_model,
                d_state=d_state,
                expand=expand_ratio,
            )
            self.active_backend = "mamba_ssm"
        else:
            self.ssm = CustomSelectiveScanSSM(
                d_model=d_model,
                d_state=d_state,
                expand=expand_ratio,
            )
            self.active_backend = "custom_ssm"

        if num_scan_directions == 4 and scan_merge == "concat":
            self.concat_proj = nn.Linear(4 * d_model, d_model, bias=False)
        else:
            self.concat_proj = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)

    def _generate_4_directional_sequences(
        self, x: torch.Tensor, H: int, W: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        x_raster = x
        x_rev_raster = torch.flip(x, dims=[1])
        x_spatial = x.view(B, H, W, D)
        x_col = x_spatial.transpose(1, 2).contiguous().view(B, L, D)
        x_rev_col = torch.flip(x_col, dims=[1])
        return x_raster, x_rev_raster, x_col, x_rev_col

    def _restore_4_directional_sequences(
        self,
        y_raster: torch.Tensor,
        y_rev_raster: torch.Tensor,
        y_col: torch.Tensor,
        y_rev_col: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = y_raster.shape
        y1 = y_raster
        y2 = torch.flip(y_rev_raster, dims=[1])
        y3 = (
            y_col.view(B, W, H, D)
            .transpose(1, 2)
            .contiguous()
            .view(B, L, D)
        )
        y4 = (
            torch.flip(y_rev_col, dims=[1])
            .view(B, W, H, D)
            .transpose(1, 2)
            .contiguous()
            .view(B, L, D)
        )
        return y1, y2, y3, y4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3 and x.shape[-1] == self.d_model
        B, L, D = x.shape
        H, W = self.spatial_size
        assert L == H * W

        residual = x
        x_norm = self.norm(x)

        if self.num_scan_directions == 1:
            y = self.ssm(x_norm)
        else:
            x1, x2, x3, x4 = self._generate_4_directional_sequences(x_norm, H, W)
            xs = torch.cat([x1, x2, x3, x4], dim=0)
            ys = self.ssm(xs)
            y1_raw, y2_raw, y3_raw, y4_raw = torch.chunk(ys, chunks=4, dim=0)
            y1, y2, y3, y4 = self._restore_4_directional_sequences(
                y1_raw, y2_raw, y3_raw, y4_raw, H, W
            )

            if self.scan_merge == "average":
                y = (y1 + y2 + y3 + y4) / 4.0
            else:
                y = self.concat_proj(torch.cat([y1, y2, y3, y4], dim=-1))

        out = residual + self.dropout(y)
        assert out.shape == (B, L, D)
        return out


class VisionMambaEncoder(nn.Module):
    """Vision Mamba Context Encoder composed of stacked VisionMambaBlocks."""

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        expand_ratio: int = 2,
        dropout: float = 0.1,
        num_blocks: int = 4,
        backend: Literal["mamba_ssm", "custom_ssm"] = "custom_ssm",
        num_scan_directions: int = 4,
        scan_merge: Literal["average", "concat"] = "average",
        spatial_size: Tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.spatial_size = spatial_size
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList([
            VisionMambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand_ratio=expand_ratio,
                dropout=dropout,
                backend=backend,
                num_scan_directions=num_scan_directions,
                scan_merge=scan_merge,
                spatial_size=spatial_size,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        B, C, H, W = x.shape
        assert C == self.d_model
        assert (H, W) == self.spatial_size

        tokens = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        for block in self.blocks:
            tokens = block(tokens)

        out = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        assert out.shape == (B, self.d_model, H, W)
        return out


if __name__ == "__main__":
    print("Testing VisionMambaEncoder...")
    encoder = VisionMambaEncoder(
        d_model=128,
        d_state=16,
        expand_ratio=2,
        dropout=0.1,
        num_blocks=1,
        backend="custom_ssm",
        num_scan_directions=4,
    )
    dummy = torch.randn(2, 128, 64, 64)
    out = encoder(dummy)
    assert out.shape == (2, 128, 64, 64)
    print("VisionMambaEncoder verified successfully!")
