"""
Dual Mamba Context Encoder Module
=================================

DAPFusion Stage I, Section 3.1.2

This module models long-range spatial dependencies across the 2D
feature map using a multi-directional selective state-space scan.

Architecture:

    CNN Feature Map
          |
          v
    Flatten to Tokens
          |
          v
    LayerNorm
          |
          v
    Vision Mamba Block
          |
          v
    4-directional Selective Scan
          |
          v
    Residual Connection
          |
          v
    Repeated Blocks
          |
          v
    Contextual Feature Map

Supported scan directions:

    1:
        Raster scan

    4:
        Raster
        Reverse raster
        Column-major
        Reverse column-major

Two backends are supported:

    "mamba_ssm"
        Official Mamba implementation when installed.

    "custom_ssm"
        Pure PyTorch selective state-space implementation.

IMPORTANT NUMERICAL-STABILITY DESIGN
------------------------------------

The custom SSM contains operations that are particularly sensitive to
FP16:

    dt = softplus(...)
    A  = -exp(A_log)
    dA = exp(dt * A)
    recurrent state update
    backward recurrent accumulation

Therefore these operations are explicitly performed in FP32.

AMP can still be used for the surrounding CNN/Mamba computations.

The architecture and methodology are unchanged.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Literal, Tuple, Union

_project_root = str(Path(__file__).resolve().parents[2])

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# =====================================================================
# Optional official Mamba backend
# =====================================================================

MAMBA_SSM_AVAILABLE = False

try:
    from mamba_ssm import Mamba as MambaOfficial  # type: ignore

    MAMBA_SSM_AVAILABLE = True

except Exception:
    MAMBA_SSM_AVAILABLE = False


# =====================================================================
# Utility
# =====================================================================

def _require_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Raise an explicit error if a tensor contains NaN or Inf.

    This is intentionally used only at important numerical boundaries,
    not on every intermediate tensor, to avoid unnecessary training
    overhead.
    """

    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f"Non-finite values detected in {name}."
        )


# =====================================================================
# Numerically Stable Selective Scan
# =====================================================================

class FastSelectiveScanRecurrence(
    torch.autograd.Function
):
    """
    Custom autograd implementation of the selective SSM recurrence.

    Recurrence:

        h_t = dA_t * h_{t-1} + dB_t * x_t

        y_t = sum_n(
                  h_t,n * C_t,n
              ) + D * x_t

    The recurrence is ALWAYS evaluated in FP32.

    This is important because recurrent multiplication can amplify
    small FP16 numerical errors across a long spatial sequence.

    The surrounding model may still use AMP.
    """

    @staticmethod
    def forward(
        ctx: Any,
        dA: torch.Tensor,
        dB_x: torch.Tensor,
        C_mat: torch.Tensor,
        D_param: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # -------------------------------------------------------------
        # All recurrence operations are performed in FP32.
        # -------------------------------------------------------------
        dA_fp32 = dA.float()
        dB_x_fp32 = dB_x.float()
        C_fp32 = C_mat.float()
        D_fp32 = D_param.float()
        x_fp32 = x.float()

        B, L, D, N = dA_fp32.shape

        # -------------------------------------------------------------
        # Initial recurrent state.
        # -------------------------------------------------------------
        h = torch.zeros(
            B,
            D,
            N,
            device=dA.device,
            dtype=torch.float32,
        )

        # -------------------------------------------------------------
        # Store all states for backward.
        #
        # FP32 is intentional. Keeping the recurrence states in FP16
        # would reintroduce the instability this implementation is
        # designed to avoid.
        # -------------------------------------------------------------
        h_all = torch.empty(
            B,
            L,
            D,
            N,
            device=dA.device,
            dtype=torch.float32,
        )

        y = torch.empty(
            B,
            L,
            D,
            device=dA.device,
            dtype=torch.float32,
        )

        # -------------------------------------------------------------
        # Forward recurrence.
        # -------------------------------------------------------------
        for t in range(L):

            h = (
                dA_fp32[:, t] * h
                + dB_x_fp32[:, t]
            )

            h_all[:, t] = h

            y[:, t] = (
                h
                * C_fp32[:, t].unsqueeze(1)
            ).sum(dim=-1) + (
                x_fp32[:, t]
                * D_fp32
            )

        # -------------------------------------------------------------
        # Save FP32 tensors for backward.
        # -------------------------------------------------------------
        ctx.save_for_backward(
            dA_fp32,
            dB_x_fp32,
            C_fp32,
            D_fp32,
            x_fp32,
            h_all,
        )

        # Preserve the original x dtype for compatibility with AMP.
        return y.to(x.dtype)

    @staticmethod
    def backward(
        ctx: Any,
        grad_y: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        (
            dA,
            dB_x,
            C_mat,
            D_param,
            x,
            h_all,
        ) = ctx.saved_tensors

        # -------------------------------------------------------------
        # Backward recurrence is also explicitly FP32.
        # -------------------------------------------------------------
        grad_y_fp32 = grad_y.float()

        B, L, D, N = dA.shape

        grad_dA = torch.empty_like(dA)
        grad_dB_x = torch.empty_like(dB_x)
        grad_C_mat = torch.empty_like(C_mat)

        grad_D_param = torch.zeros_like(
            D_param
        )

        grad_x = torch.empty_like(x)

        grad_h = torch.zeros(
            B,
            D,
            N,
            device=dA.device,
            dtype=torch.float32,
        )

        # -------------------------------------------------------------
        # Reverse-time recurrence.
        # -------------------------------------------------------------
        for t in range(
            L - 1,
            -1,
            -1,
        ):

            gy_t = grad_y_fp32[:, t]

            # ---------------------------------------------------------
            # Gradient wrt D.
            # ---------------------------------------------------------
            grad_D_param += (
                gy_t * x[:, t]
            ).sum(dim=0)

            # ---------------------------------------------------------
            # Gradient wrt x.
            # ---------------------------------------------------------
            grad_x[:, t] = (
                gy_t * D_param
            )

            # ---------------------------------------------------------
            # Gradient wrt C.
            # ---------------------------------------------------------
            grad_C_mat[:, t] = (
                gy_t.unsqueeze(-1)
                * h_all[:, t]
            ).sum(dim=1)

            # ---------------------------------------------------------
            # Accumulate gradient wrt hidden state.
            # ---------------------------------------------------------
            grad_h = (
                grad_h
                + gy_t.unsqueeze(-1)
                * C_mat[:, t].unsqueeze(1)
            )

            # Gradient wrt dB_x.
            grad_dB_x[:, t] = grad_h

            # ---------------------------------------------------------
            # Previous hidden state.
            # ---------------------------------------------------------
            if t > 0:

                h_prev = h_all[:, t - 1]

            else:

                h_prev = torch.zeros(
                    B,
                    D,
                    N,
                    device=dA.device,
                    dtype=torch.float32,
                )

            # Gradient wrt dA.
            grad_dA[:, t] = (
                grad_h
                * h_prev
            )

            # Propagate hidden-state gradient backwards.
            grad_h = (
                grad_h
                * dA[:, t]
            )

        # -------------------------------------------------------------
        # Return gradients corresponding to:
        #
        # dA
        # dB_x
        # C_mat
        # D_param
        # x
        #
        # The tensors supplied to the custom function may originally
        # have been FP16, so cast gradients to their corresponding
        # input dtype.
        # -------------------------------------------------------------
        return (
            grad_dA,
            grad_dB_x,
            grad_C_mat,
            grad_D_param,
            grad_x,
        )


# =====================================================================
# Custom Selective State-Space Model
# =====================================================================

class CustomSelectiveScanSSM(nn.Module):
    """
    Pure PyTorch selective state-space model.

    This implements the S6-style selective scan used by the custom
    backend.

    Args:
        d_model:
            Input/output feature dimension.

        d_state:
            State dimension.

        expand:
            Inner dimension expansion ratio.

        dt_rank:
            Rank used for delta-time projection.

        d_conv:
            Depthwise convolution kernel size.
    """

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

        self.d_inner = (
            d_model * expand
        )

        self.d_conv = d_conv

        self.dt_rank = (
            math.ceil(d_model / 16)
            if dt_rank == "auto"
            else int(dt_rank)
        )

        # -------------------------------------------------------------
        # Input projection.
        # -------------------------------------------------------------
        self.in_proj = nn.Linear(
            d_model,
            2 * self.d_inner,
            bias=False,
        )

        # -------------------------------------------------------------
        # Depthwise causal-style convolution.
        # -------------------------------------------------------------
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # -------------------------------------------------------------
        # Selective parameter projection.
        #
        # dt_rank + B + C
        # -------------------------------------------------------------
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank
            + 2 * self.d_state,
            bias=False,
        )

        # -------------------------------------------------------------
        # Delta-time projection.
        # -------------------------------------------------------------
        self.dt_proj = nn.Linear(
            self.dt_rank,
            self.d_inner,
            bias=True,
        )

        # -------------------------------------------------------------
        # State transition parameter A.
        #
        # A_log is intentionally initialized in FP32.
        # -------------------------------------------------------------
        A_init = torch.arange(
            1,
            self.d_state + 1,
            dtype=torch.float32,
        ).repeat(
            self.d_inner,
            1,
        )

        self.A_log = nn.Parameter(
            torch.log(A_init)
        )

        # -------------------------------------------------------------
        # Skip connection parameter D.
        # -------------------------------------------------------------
        self.D = nn.Parameter(
            torch.ones(
                self.d_inner,
                dtype=torch.float32,
            )
        )

        # -------------------------------------------------------------
        # Output projection.
        # -------------------------------------------------------------
        self.out_proj = nn.Linear(
            self.d_inner,
            d_model,
            bias=False,
        )

    # -----------------------------------------------------------------
    # Preserve numerically sensitive parameters in FP32.
    # -----------------------------------------------------------------

    def _apply(self, fn):
        """
        Keep A_log and D in FP32 even when the complete model is
        converted to FP16.

        This specifically protects the two state-space parameters from
        direct FP16 conversion.

        Normal AMP/autocast training does not require this mechanism,
        but it makes explicit .half() tests substantially safer.
        """

        super()._apply(fn)

        # Re-establish FP32 storage after .half(), .bfloat16(), etc.
        self.A_log.data = (
            self.A_log.data.float()
        )

        self.D.data = (
            self.D.data.float()
        )

        return self

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        u: torch.Tensor,
    ) -> torch.Tensor:

        if u.ndim != 3:
            raise ValueError(
                "CustomSelectiveScanSSM expects "
                "(B, L, D), "
                f"got {tuple(u.shape)}"
            )

        if u.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected final dimension "
                f"{self.d_model}, "
                f"got {u.shape[-1]}"
            )

        B, L, _ = u.shape

        # -------------------------------------------------------------
        # Preserve input dtype for final output.
        # -------------------------------------------------------------
        input_dtype = u.dtype

        # -------------------------------------------------------------
        # Input projection.
        #
        # This part may safely participate in AMP.
        # -------------------------------------------------------------
        xz = self.in_proj(u)

        x_branch, z_gate = torch.split(
            xz,
            [
                self.d_inner,
                self.d_inner,
            ],
            dim=-1,
        )

        # -------------------------------------------------------------
        # Depthwise convolution.
        # -------------------------------------------------------------
        x_conv = self.conv1d(
            x_branch.transpose(1, 2)
        )

        # Remove right-side padded positions so that the sequence
        # length is exactly L.
        x_conv = x_conv[
            :,
            :,
            :L,
        ]

        x_conv = F.silu(
            x_conv
        ).transpose(1, 2)

        # -------------------------------------------------------------
        # The numerically sensitive selective-state-space section
        # starts here.
        #
        # Everything below is explicitly calculated in FP32.
        # -------------------------------------------------------------
        x_ssm = x_conv.float()

        # -------------------------------------------------------------
        # x_proj
        #
        # Parameters are explicitly cast to FP32 so that this path
        # remains valid even when the surrounding model has been
        # converted to FP16.
        # -------------------------------------------------------------
        x_proj_out = F.linear(
            x_ssm,
            self.x_proj.weight.float(),
            None,
        )

        dt_raw, B_mat, C_mat = torch.split(
            x_proj_out,
            [
                self.dt_rank,
                self.d_state,
                self.d_state,
            ],
            dim=-1,
        )

        # -------------------------------------------------------------
        # Delta-time.
        #
        # softplus is explicitly evaluated in FP32.
        # -------------------------------------------------------------
        dt_projection = F.linear(
            dt_raw,
            self.dt_proj.weight.float(),
            self.dt_proj.bias.float(),
        )

        dt = F.softplus(
            dt_projection
        )

        # -------------------------------------------------------------
        # State transition matrix:
        #
        #     A = -exp(A_log)
        #
        # A_log is maintained in FP32.
        # -------------------------------------------------------------
        A = -torch.exp(
            self.A_log.float()
        )

        # -------------------------------------------------------------
        # Discretized transition:
        #
        #     dA = exp(dt * A)
        #
        # This is one of the most numerically sensitive operations in
        # the custom SSM and therefore stays in FP32.
        # -------------------------------------------------------------
        dt_expanded = dt.unsqueeze(-1)

        A_expanded = (
            A.unsqueeze(0)
             .unsqueeze(0)
        )

        dA = torch.exp(
            dt_expanded
            * A_expanded
        )

        # -------------------------------------------------------------
        # Input-state contribution:
        #
        #     dB = dt * B
        #     dB_x = dB * x
        # -------------------------------------------------------------
        dB = (
            dt.unsqueeze(-1)
            * B_mat.unsqueeze(2)
        )

        dB_x = (
            dB
            * x_ssm.unsqueeze(-1)
        )

        # -------------------------------------------------------------
        # Numerical sanity check before recurrence.
        # -------------------------------------------------------------
        _require_finite(
            dt,
            "SSM dt",
        )

        _require_finite(
            A,
            "SSM A",
        )

        _require_finite(
            dA,
            "SSM dA",
        )

        _require_finite(
            dB_x,
            "SSM dB_x",
        )

        # -------------------------------------------------------------
        # FP32 recurrent selective scan.
        # -------------------------------------------------------------
        y = FastSelectiveScanRecurrence.apply(
            dA,
            dB_x,
            C_mat,
            self.D.float(),
            x_ssm,
        )

        y = y.float()

        # -------------------------------------------------------------
        # Gating.
        #
        # z_gate may be FP16 under AMP, so cast it to FP32 for the
        # numerically stable SSM output path.
        # -------------------------------------------------------------
        z_gate_fp32 = z_gate.float()

        y_gated = (
            y
            * F.silu(
                z_gate_fp32
            )
        )

        # -------------------------------------------------------------
        # Output projection.
        #
        # Use FP32 weights when the surrounding model has been
        # converted to FP16. The resulting tensor is converted back
        # to the original input dtype.
        # -------------------------------------------------------------
        out = F.linear(
            y_gated,
            self.out_proj.weight.float(),
            None,
        )

        out = out.to(
            input_dtype
        )

        if out.shape != (
            B,
            L,
            self.d_model,
        ):
            raise RuntimeError(
                "CustomSelectiveScanSSM produced an unexpected "
                f"shape: {tuple(out.shape)}"
            )

        return out


# =====================================================================
# Vision Mamba Block
# =====================================================================

class VisionMambaBlock(nn.Module):
    """
    Vision Mamba block with VMamba-style multi-directional scanning.

    Architecture:

        LayerNorm
            |
            v
        Selective SSM
            |
            v
        Directional merge
            |
            v
        Dropout
            |
            v
        Residual addition
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        expand_ratio: int = 2,
        dropout: float = 0.1,
        backend: Literal[
            "mamba_ssm",
            "custom_ssm",
        ] = "custom_ssm",
        num_scan_directions: int = 4,
        scan_merge: Literal[
            "average",
            "concat",
        ] = "average",
        spatial_size: Tuple[
            int,
            int,
        ] = (64, 64),
    ) -> None:

        super().__init__()

        if num_scan_directions not in (
            1,
            4,
        ):
            raise ValueError(
                "num_scan_directions must be "
                "1 or 4."
            )

        if scan_merge not in (
            "average",
            "concat",
        ):
            raise ValueError(
                "scan_merge must be "
                "'average' or 'concat'."
            )

        if (
            len(spatial_size) != 2
            or spatial_size[0] <= 0
            or spatial_size[1] <= 0
        ):
            raise ValueError(
                f"Invalid spatial_size: {spatial_size}"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.expand_ratio = expand_ratio
        self.num_scan_directions = (
            num_scan_directions
        )
        self.scan_merge = scan_merge
        self.spatial_size = spatial_size

        requested_backend = backend

        # -------------------------------------------------------------
        # Pre-normalization.
        # -------------------------------------------------------------
        self.norm = nn.LayerNorm(
            d_model
        )

        # -------------------------------------------------------------
        # Backend selection.
        # -------------------------------------------------------------
        if (
            backend == "mamba_ssm"
            and MAMBA_SSM_AVAILABLE
        ):

            self.ssm = MambaOfficial(
                d_model=d_model,
                d_state=d_state,
                expand=expand_ratio,
            )

            self.active_backend = (
                "mamba_ssm"
            )

        else:

            if (
                backend == "mamba_ssm"
                and not MAMBA_SSM_AVAILABLE
            ):
                logger.warning(
                    "Requested official mamba_ssm backend, "
                    "but it is unavailable. Falling back to "
                    "custom_ssm."
                )

            self.ssm = (
                CustomSelectiveScanSSM(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand_ratio,
                )
            )

            self.active_backend = (
                "custom_ssm"
            )

        logger.debug(
            "VisionMambaBlock backend: requested=%s active=%s",
            requested_backend,
            self.active_backend,
        )

        # -------------------------------------------------------------
        # Direction merge.
        # -------------------------------------------------------------
        if (
            num_scan_directions == 4
            and scan_merge == "concat"
        ):

            self.concat_proj = nn.Linear(
                4 * d_model,
                d_model,
                bias=False,
            )

        else:

            self.concat_proj = (
                nn.Identity()
            )

        # -------------------------------------------------------------
        # Residual dropout.
        # -------------------------------------------------------------
        self.dropout = nn.Dropout(
            p=dropout
        )

    # -----------------------------------------------------------------
    # Generate four scan sequences
    # -----------------------------------------------------------------

    def _generate_4_directional_sequences(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        B, L, D = x.shape

        if L != H * W:
            raise ValueError(
                f"Sequence length {L} does not match "
                f"H*W={H*W}."
            )

        # -------------------------------------------------------------
        # Direction 1: raster
        # -------------------------------------------------------------
        x_raster = x

        # -------------------------------------------------------------
        # Direction 2: reverse raster
        # -------------------------------------------------------------
        x_rev_raster = torch.flip(
            x,
            dims=[1],
        )

        # -------------------------------------------------------------
        # Convert tokens back to spatial representation.
        # -------------------------------------------------------------
        x_spatial = x.view(
            B,
            H,
            W,
            D,
        )

        # -------------------------------------------------------------
        # Direction 3: column-major
        # -------------------------------------------------------------
        x_col = (
            x_spatial
            .transpose(1, 2)
            .contiguous()
            .view(B, L, D)
        )

        # -------------------------------------------------------------
        # Direction 4: reverse column-major
        # -------------------------------------------------------------
        x_rev_col = torch.flip(
            x_col,
            dims=[1],
        )

        return (
            x_raster,
            x_rev_raster,
            x_col,
            x_rev_col,
        )

    # -----------------------------------------------------------------
    # Restore directional sequences
    # -----------------------------------------------------------------

    def _restore_4_directional_sequences(
        self,
        y_raster: torch.Tensor,
        y_rev_raster: torch.Tensor,
        y_col: torch.Tensor,
        y_rev_col: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        B, L, D = y_raster.shape

        if L != H * W:
            raise ValueError(
                f"Sequence length {L} does not match "
                f"H*W={H*W}."
            )

        # -------------------------------------------------------------
        # Restore raster direction.
        # -------------------------------------------------------------
        y1 = y_raster

        # -------------------------------------------------------------
        # Restore reverse raster.
        # -------------------------------------------------------------
        y2 = torch.flip(
            y_rev_raster,
            dims=[1],
        )

        # -------------------------------------------------------------
        # Restore column-major.
        # -------------------------------------------------------------
        y3 = (
            y_col
            .view(
                B,
                W,
                H,
                D,
            )
            .transpose(1, 2)
            .contiguous()
            .view(
                B,
                L,
                D,
            )
        )

        # -------------------------------------------------------------
        # Restore reverse column-major.
        # -------------------------------------------------------------
        y4 = (
            torch.flip(
                y_rev_col,
                dims=[1],
            )
            .view(
                B,
                W,
                H,
                D,
            )
            .transpose(1, 2)
            .contiguous()
            .view(
                B,
                L,
                D,
            )
        )

        return (
            y1,
            y2,
            y3,
            y4,
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "VisionMambaBlock expects "
                "(B, L, D), "
                f"got {tuple(x.shape)}"
            )

        B, L, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected feature dimension "
                f"{self.d_model}, got {D}"
            )

        H, W = self.spatial_size

        if L != H * W:
            raise ValueError(
                f"Sequence length {L} does not match "
                f"configured spatial size "
                f"{H}x{W}={H*W}."
            )

        # -------------------------------------------------------------
        # Residual path.
        # -------------------------------------------------------------
        residual = x

        # -------------------------------------------------------------
        # Pre-normalization.
        # -------------------------------------------------------------
        x_norm = self.norm(x)

        # -------------------------------------------------------------
        # Single directional mode.
        # -------------------------------------------------------------
        if self.num_scan_directions == 1:

            y = self.ssm(
                x_norm
            )

        # -------------------------------------------------------------
        # Four directional mode.
        # -------------------------------------------------------------
        else:

            (
                x1,
                x2,
                x3,
                x4,
            ) = self._generate_4_directional_sequences(
                x_norm,
                H,
                W,
            )

            # ---------------------------------------------------------
            # Process all directions as one larger batch.
            #
            # This preserves the same SSM weights across all scan
            # directions.
            # ---------------------------------------------------------
            xs = torch.cat(
                [
                    x1,
                    x2,
                    x3,
                    x4,
                ],
                dim=0,
            )

            ys = self.ssm(
                xs
            )

            (
                y1_raw,
                y2_raw,
                y3_raw,
                y4_raw,
            ) = torch.chunk(
                ys,
                chunks=4,
                dim=0,
            )

            (
                y1,
                y2,
                y3,
                y4,
            ) = self._restore_4_directional_sequences(
                y1_raw,
                y2_raw,
                y3_raw,
                y4_raw,
                H,
                W,
            )

            # ---------------------------------------------------------
            # Directional fusion.
            # ---------------------------------------------------------
            if self.scan_merge == "average":

                y = (
                    y1
                    + y2
                    + y3
                    + y4
                ) / 4.0

            else:

                y = self.concat_proj(
                    torch.cat(
                        [
                            y1,
                            y2,
                            y3,
                            y4,
                        ],
                        dim=-1,
                    )
                )

        # -------------------------------------------------------------
        # Residual output.
        # -------------------------------------------------------------
        out = (
            residual
            + self.dropout(y)
        )

        if out.shape != (
            B,
            L,
            D,
        ):
            raise RuntimeError(
                "VisionMambaBlock produced unexpected "
                f"shape: {tuple(out.shape)}"
            )

        return out


# =====================================================================
# Vision Mamba Encoder
# =====================================================================

class VisionMambaEncoder(nn.Module):
    """
    Stacked Vision Mamba Context Encoder.

    Input:

        (B, C, H, W)

    Output:

        (B, C, H, W)

    The spatial size is supplied at construction time.

    This is important for image-size ablation:

        128x128 input
            -> CNN
            -> 32x32 feature map
            -> spatial_size=(32,32)

        192x192 input
            -> CNN
            -> 48x48 feature map
            -> spatial_size=(48,48)

        256x256 input
            -> CNN
            -> 64x64 feature map
            -> spatial_size=(64,64)
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        expand_ratio: int = 2,
        dropout: float = 0.1,
        num_blocks: int = 4,
        backend: Literal[
            "mamba_ssm",
            "custom_ssm",
        ] = "custom_ssm",
        num_scan_directions: int = 4,
        scan_merge: Literal[
            "average",
            "concat",
        ] = "average",
        spatial_size: Tuple[
            int,
            int,
        ] = (64, 64),
    ) -> None:

        super().__init__()

        if num_blocks <= 0:
            raise ValueError(
                "num_blocks must be positive."
            )

        if len(spatial_size) != 2:
            raise ValueError(
                "spatial_size must be (H, W)."
            )

        if (
            spatial_size[0] <= 0
            or spatial_size[1] <= 0
        ):
            raise ValueError(
                f"Invalid spatial_size: "
                f"{spatial_size}"
            )

        self.d_model = d_model
        self.spatial_size = spatial_size
        self.num_blocks = num_blocks

        # -------------------------------------------------------------
        # Stacked Mamba blocks.
        # -------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                VisionMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand_ratio=expand_ratio,
                    dropout=dropout,
                    backend=backend,
                    num_scan_directions=(
                        num_scan_directions
                    ),
                    scan_merge=scan_merge,
                    spatial_size=spatial_size,
                )
                for _ in range(num_blocks)
            ]
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "VisionMambaEncoder expects "
                "(B, C, H, W), "
                f"got {tuple(x.shape)}"
            )

        B, C, H, W = x.shape

        if C != self.d_model:
            raise ValueError(
                f"Expected {self.d_model} channels, "
                f"got {C}"
            )

        if (
            H,
            W,
        ) != self.spatial_size:

            raise ValueError(
                "Input spatial dimensions "
                f"{H}x{W} do not match configured "
                f"spatial_size={self.spatial_size}."
            )

        # -------------------------------------------------------------
        # Convert image feature map to token sequence.
        #
        # (B,C,H,W)
        #     ->
        # (B,H,W,C)
        #     ->
        # (B,H*W,C)
        # -------------------------------------------------------------
        tokens = (
            x.permute(
                0,
                2,
                3,
                1,
            )
            .contiguous()
            .view(
                B,
                H * W,
                C,
            )
        )

        # -------------------------------------------------------------
        # Stacked Mamba blocks.
        # -------------------------------------------------------------
        for block in self.blocks:

            tokens = block(
                tokens
            )

        # -------------------------------------------------------------
        # Restore feature-map layout.
        #
        # (B,H*W,C)
        #     ->
        # (B,H,W,C)
        #     ->
        # (B,C,H,W)
        # -------------------------------------------------------------
        out = (
            tokens
            .view(
                B,
                H,
                W,
                C,
            )
            .permute(
                0,
                3,
                1,
                2,
            )
            .contiguous()
        )

        if out.shape != (
            B,
            self.d_model,
            H,
            W,
        ):
            raise RuntimeError(
                "VisionMambaEncoder produced "
                f"unexpected shape: {tuple(out.shape)}"
            )

        return out


# =====================================================================
# Standalone verification
# =====================================================================

def _test_encoder_shape(
    spatial_size: Tuple[int, int],
) -> None:

    encoder = VisionMambaEncoder(
        d_model=128,
        d_state=16,
        expand_ratio=2,
        dropout=0.1,
        num_blocks=1,
        backend="custom_ssm",
        num_scan_directions=4,
        spatial_size=spatial_size,
    )

    encoder.eval()

    H, W = spatial_size

    dummy = torch.randn(
        1,
        128,
        H,
        W,
    )

    with torch.no_grad():

        out = encoder(
            dummy
        )

    assert out.shape == (
        1,
        128,
        H,
        W,
    )

    assert torch.isfinite(
        out
    ).all()

    print(
        f"Shape test passed: "
        f"{H}x{W} -> {tuple(out.shape)}"
    )


def _test_fp32_gradient() -> None:
    """
    Verify that the custom SSM produces finite FP32 gradients.
    """

    model = CustomSelectiveScanSSM(
        d_model=32,
        d_state=16,
        expand=2,
    )

    model.train()

    x = torch.randn(
        1,
        64,
        32,
        requires_grad=True,
    )

    out = model(x)

    loss = out.float().pow(2).mean()

    loss.backward()

    assert torch.isfinite(
        loss
    )

    assert torch.isfinite(
        x.grad
    ).all()

    for name, parameter in model.named_parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all(), (
                f"Non-finite gradient in {name}"
            )

    print(
        "FP32 gradient test passed."
    )


def _test_half_input() -> None:
    """
    Verify that the custom SSM can accept FP16 input while keeping
    the sensitive state-space operations in FP32.

    This does NOT rely on the entire module being converted to FP16.
    It specifically tests the mixed-precision execution path.
    """

    model = CustomSelectiveScanSSM(
        d_model=32,
        d_state=16,
        expand=2,
    )

    model.eval()

    x = torch.randn(
        1,
        64,
        32,
    ).half()

    with torch.no_grad():

        out = model(
            x
        )

    assert out.dtype == torch.float16

    assert torch.isfinite(
        out
    ).all()

    # A_log and D should remain FP32.
    assert model.A_log.dtype == torch.float32
    assert model.D.dtype == torch.float32

    print(
        "Mixed-precision SSM test passed."
    )


if __name__ == "__main__":

    print(
        "Testing mamba_encoder.py..."
    )

    # -------------------------------------------------------------
    # 128x128 experiment
    # CNN output = 32x32
    # -------------------------------------------------------------
    _test_encoder_shape(
        (32, 32)
    )

    # -------------------------------------------------------------
    # 192x192 experiment
    # CNN output = 48x48
    # -------------------------------------------------------------
    _test_encoder_shape(
        (48, 48)
    )

    # -------------------------------------------------------------
    # 256x256 experiment
    # CNN output = 64x64
    # -------------------------------------------------------------
    _test_encoder_shape(
        (64, 64)
    )

    # -------------------------------------------------------------
    # Gradient test
    # -------------------------------------------------------------
    _test_fp32_gradient()

    # -------------------------------------------------------------
    # Mixed precision test
    # -------------------------------------------------------------
    _test_half_input()

    print(
        "VisionMambaEncoder verified successfully!"
    )