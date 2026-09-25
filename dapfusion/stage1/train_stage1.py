"""
Stage I Training Pipeline
=========================

DAPFusion Framework - Stage I:
Multi-Modal Feature Representation Backbone

Training pipeline:

    RGB + Thermal
          |
          v
      Stage1Model
          |
          v
      Stage1Loss
          |
          v
       AdamW
          |
          v
    Stage-I checkpoint

The four Stage-I objectives remain unchanged:

    L_structure
    L_decorr
    L_comp
    L_info

with:

    L_total =
        lambda_structure * L_structure
        + lambda_decorr * L_decorr
        + lambda_comp * L_comp
        + lambda_info * L_info

Training features:

    - AdamW
    - AMP on CUDA
    - Gradient scaling
    - Gradient accumulation
    - Gradient clipping
    - NaN / Inf detection
    - Checkpoint saving
    - Checkpoint resume
    - Configurable image size
    - LLVIP integration
    - Synthetic dry run
    - Real-data smoke testing
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

# =====================================================================
# Project root
# =====================================================================

_project_root = str(
    Path(__file__).resolve().parents[2]
)

if _project_root not in sys.path:
    sys.path.insert(
        0,
        _project_root,
    )


# =====================================================================
# Imports
# =====================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from dapfusion.stage1.config import Stage1Config
from dapfusion.stage1.losses import Stage1Loss
from dapfusion.stage1.stage1_model import Stage1Model


# =====================================================================
# Logging
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "[%(levelname)s] "
        "%(message)s"
    ),
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
)

logger = logging.getLogger(
    "TrainStage1"
)


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------
    # cuDNN settings.
    #
    # benchmark=False improves reproducibility.
    # -------------------------------------------------------------
    if torch.backends.cudnn.is_available():

        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = True


# =====================================================================
# Device handling
# =====================================================================

def resolve_device(
    requested_device: Optional[
        Union[str, torch.device]
    ],
) -> torch.device:
    """
    Resolve:

        "auto"
        "cuda"
        "cpu"

    into an actual torch.device.
    """

    if requested_device is None:

        requested = "auto"

    else:

        requested = str(
            requested_device
        ).lower()

    # -------------------------------------------------------------
    # Automatic selection
    # -------------------------------------------------------------
    if requested == "auto":

        if torch.cuda.is_available():

            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    # -------------------------------------------------------------
    # Explicit CUDA request
    # -------------------------------------------------------------
    if requested.startswith(
        "cuda"
    ):

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested, but "
                "CUDA is not available."
            )

        return torch.device(
            requested
        )

    # -------------------------------------------------------------
    # CPU
    # -------------------------------------------------------------
    if requested == "cpu":

        return torch.device(
            "cpu"
        )

    raise ValueError(
        f"Unsupported device: "
        f"{requested_device}"
    )


# =====================================================================
# AMP helpers
# =====================================================================

def amp_enabled(
    device: torch.device,
    config: Stage1Config,
) -> bool:
    """
    Return whether AMP should be enabled.
    """

    return (
        config.use_amp
        and device.type == "cuda"
        and torch.cuda.is_available()
    )


def create_grad_scaler(
    enabled: bool,
):
    """
    Create the CUDA gradient scaler.

    The fallback syntax keeps compatibility with different
    PyTorch versions.
    """

    try:

        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )

    except (
        AttributeError,
        TypeError,
    ):

        return torch.cuda.amp.GradScaler(
            enabled=enabled
        )


class NullScaler:
    """
    Minimal scaler interface used when AMP is disabled.

    This avoids branching throughout the training loop.
    """

    def scale(
        self,
        loss: torch.Tensor,
    ) -> torch.Tensor:

        return loss

    def unscale_(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:

        return None

    def step(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:

        optimizer.step()

    def update(
        self,
    ) -> None:

        return None

    def state_dict(
        self,
    ) -> Dict:

        return {}

    def load_state_dict(
        self,
        state_dict: Dict,
    ) -> None:

        return None


# =====================================================================
# Autocast context helper
# =====================================================================

def autocast_context(
    enabled: bool,
):
    """
    Return an appropriate CUDA autocast context.

    The model's numerically sensitive custom SSM operations explicitly
    switch to FP32 internally.
    """

    if not enabled:

        return torch.autocast(
            device_type="cpu",
            enabled=False,
        )

    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    )


# =====================================================================
# Gradient diagnostics
# =====================================================================

def check_gradients(
    model: nn.Module,
) -> Tuple[
    bool,
    float,
]:
    """
    Check all model gradients for NaN / Inf.

    Returns:

        gradients_are_finite
        total_gradient_norm
    """

    total_sq_norm = 0.0

    gradients_are_finite = True

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        gradient = parameter.grad

        if not torch.isfinite(
            gradient
        ).all():

            gradients_are_finite = False

            logger.error(
                "Non-finite gradient detected "
                f"in parameter: {name}"
            )

            continue

        param_norm = (
            gradient.detach()
            .float()
            .norm(2)
            .item()
        )

        total_sq_norm += (
            param_norm ** 2
        )

    total_norm = math.sqrt(
        total_sq_norm
    )

    return (
        gradients_are_finite,
        total_norm,
    )


# =====================================================================
# CUDA memory logging
# =====================================================================

def log_cuda_memory(
    prefix: str = "",
) -> None:
    """
    Log current CUDA memory statistics.
    """

    if not torch.cuda.is_available():
        return

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    max_allocated = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    logger.info(
        f"{prefix}"
        f"CUDA memory | "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"peak={max_allocated:.2f} GB"
    )


# =====================================================================
# Single training step
# =====================================================================

def train_step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    device: torch.device,
    scaler=None,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    accumulation_index: int = 0,
    grad_clip_norm: Optional[float] = 1.0,
    check_gradients_enabled: bool = True,
) -> Dict[str, float]:
    """
    Perform one training step.

    Gradient accumulation is handled here.

    ``accumulation_index`` is zero-based within the current
    epoch.
    """

    model.train()

    # -------------------------------------------------------------
    # Move inputs to device.
    # -------------------------------------------------------------
    rgb = rgb.to(
        device,
        non_blocking=(
            device.type == "cuda"
        ),
    )

    thermal = thermal.to(
        device,
        non_blocking=(
            device.type == "cuda"
        ),
    )

    # -------------------------------------------------------------
    # Forward pass.
    # -------------------------------------------------------------
    with autocast_context(
        enabled=use_amp
    ):

        model_outputs = model(
            rgb,
            thermal,
        )

        losses = criterion(
            model_outputs
        )

        total_loss = losses[
            "loss_total"
        ]

    # -------------------------------------------------------------
    # Explicit loss validation.
    # -------------------------------------------------------------
    if not torch.isfinite(
        total_loss
    ):

        raise FloatingPointError(
            "Non-finite loss encountered: "
            f"{total_loss.detach().item()}"
        )

    # -------------------------------------------------------------
    # Scale loss for gradient accumulation.
    # -------------------------------------------------------------
    scaled_loss = (
        total_loss
        / float(
            gradient_accumulation_steps
        )
    )

    # -------------------------------------------------------------
    # Backward.
    # -------------------------------------------------------------
    scaler.scale(
        scaled_loss
    ).backward()

    # -------------------------------------------------------------
    # Only update the optimizer at the end of an accumulation
    # window.
    # -------------------------------------------------------------
    should_step = (
        (
            accumulation_index + 1
        )
        % gradient_accumulation_steps
        == 0
    )

    gradient_norm = 0.0

    if should_step:

        # ---------------------------------------------------------
        # Unscale gradients before clipping and checking them.
        # ---------------------------------------------------------
        scaler.unscale_(
            optimizer
        )

        if check_gradients_enabled:

            finite, gradient_norm = (
                check_gradients(
                    model
                )
            )

            if not finite:

                optimizer.zero_grad(
                    set_to_none=True
                )

                raise FloatingPointError(
                    "Non-finite gradients detected. "
                    "Optimizer step aborted."
                )

        # ---------------------------------------------------------
        # Gradient clipping.
        # ---------------------------------------------------------
        if (
            grad_clip_norm is not None
            and grad_clip_norm > 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
            )

            # Re-check after clipping.
            if check_gradients_enabled:

                finite_after_clip, gradient_norm = (
                    check_gradients(
                        model
                    )
                )

                if not finite_after_clip:

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    raise FloatingPointError(
                        "Non-finite gradients detected "
                        "after gradient clipping."
                    )

        # ---------------------------------------------------------
        # Optimizer step.
        # ---------------------------------------------------------
        scaler.step(
            optimizer
        )

        scaler.update()

        optimizer.zero_grad(
            set_to_none=True
        )

    # -------------------------------------------------------------
    # Return detached scalar losses.
    # -------------------------------------------------------------
    result = {
        key: float(
            value.detach()
            .float()
            .item()
        )
        for key, value in losses.items()
    }

    result[
        "grad_norm"
    ] = float(
        gradient_norm
    )

    result[
        "optimizer_step"
    ] = float(
        should_step
    )

    return result


# =====================================================================
# Train one epoch
# =====================================================================

def train_epoch(
    model: nn.Module,
    dataloader: Iterable[
        Tuple[
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    scaler=None,
    use_amp: bool = False,
    log_interval: int = 10,
    max_steps: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    grad_clip_norm: Optional[float] = 1.0,
    check_gradients_enabled: bool = True,
) -> Dict[str, float]:
    """
    Train for one epoch.
    """

    if scaler is None:

        scaler = NullScaler()

    running_losses: Dict[
        str,
        float,
    ] = {}

    step_count = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    epoch_t0 = time.time()

    for step, batch in enumerate(
        dataloader
    ):

        if (
            max_steps is not None
            and step >= max_steps
        ):
            break

        # ---------------------------------------------------------
        # Dataset batch validation.
        # ---------------------------------------------------------
        if not isinstance(
            batch,
            (tuple, list),
        ):

            raise TypeError(
                "Dataloader must yield "
                "(rgb, thermal)."
            )

        if len(batch) < 2:

            raise ValueError(
                "Dataloader batch must contain "
                "RGB and Thermal tensors."
            )

        rgb, thermal = batch[:2]

        step_t0 = time.time()

        # ---------------------------------------------------------
        # Training step.
        # ---------------------------------------------------------
        step_losses = train_step(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            rgb=rgb,
            thermal=thermal,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
            accumulation_index=step,
            grad_clip_norm=grad_clip_norm,
            check_gradients_enabled=(
                check_gradients_enabled
            ),
        )

        step_time = (
            time.time()
            - step_t0
        )

        step_count += 1

        # ---------------------------------------------------------
        # Running averages.
        # ---------------------------------------------------------
        for key, value in step_losses.items():

            if key in (
                "optimizer_step",
            ):
                continue

            running_losses[key] = (
                running_losses.get(
                    key,
                    0.0,
                )
                + value
            )

        # ---------------------------------------------------------
        # Logging.
        # ---------------------------------------------------------
        if (
            (step + 1) % log_interval == 0
            or (
                max_steps is not None
                and (step + 1) == max_steps
            )
        ):

            logger.info(
                f"Epoch [{epoch:03d}] "
                f"Step [{step + 1:05d}] | "
                f"Total={step_losses['loss_total']:.6f} | "
                f"Struct={step_losses['loss_structure']:.6f} | "
                f"Decorr={step_losses['loss_decorr']:.6f} | "
                f"Comp={step_losses['loss_comp']:.6f} | "
                f"Info={step_losses['loss_info']:.6f} | "
                f"GradNorm={step_losses['grad_norm']:.4f} | "
                f"{step_time:.2f}s/it"
            )

    # -------------------------------------------------------------
    # Handle a final incomplete gradient-accumulation window.
    #
    # If the number of batches is not divisible by accumulation
    # steps, the final gradients would otherwise never be stepped.
    # -------------------------------------------------------------
    if (
        step_count > 0
        and (
            step_count
            % gradient_accumulation_steps
        ) != 0
    ):

        scaler.unscale_(
            optimizer
        )

        if (
            grad_clip_norm is not None
            and grad_clip_norm > 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        optimizer.zero_grad(
            set_to_none=True
        )

    # -------------------------------------------------------------
    # Epoch averages.
    # -------------------------------------------------------------
    averages = {
        key: value
        / max(
            1,
            step_count,
        )
        for key, value in running_losses.items()
    }

    averages[
        "epoch_time"
    ] = time.time() - epoch_t0

    averages[
        "steps"
    ] = float(
        step_count
    )

    return averages


# =====================================================================
# Checkpoint saving
# =====================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    losses: Dict[str, float],
    checkpoint_dir: Union[
        str,
        Path,
    ],
    config: Stage1Config,
    scaler=None,
) -> Path:
    """
    Save a complete Stage-I checkpoint.

    Stores:

        - epoch
        - model state
        - optimizer state
        - AMP scaler state
        - losses
        - configuration
    """

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Configuration serialization.
    # -------------------------------------------------------------
    if hasattr(
        config,
        "to_dict",
    ):

        config_dict = (
            config.to_dict()
        )

    else:

        config_dict = vars(
            config
        )

    checkpoint = {
        "epoch": int(
            epoch
        ),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scaler_state_dict":
            (
                scaler.state_dict()
                if scaler is not None
                else {}
            ),

        "losses":
            dict(losses),

        "config":
            config_dict,
    }

    # -------------------------------------------------------------
    # Epoch-specific checkpoint.
    # -------------------------------------------------------------
    epoch_path = (
        checkpoint_dir
        / f"stage1_epoch_{epoch:03d}.pth"
    )

    torch.save(
        checkpoint,
        epoch_path,
    )

    # -------------------------------------------------------------
    # Latest checkpoint.
    # -------------------------------------------------------------
    latest_path = (
        checkpoint_dir
        / "latest.pth"
    )

    torch.save(
        checkpoint,
        latest_path,
    )

    logger.info(
        f"Checkpoint saved: "
        f"{epoch_path}"
    )

    logger.info(
        f"Latest checkpoint updated: "
        f"{latest_path}"
    )

    return latest_path


# =====================================================================
# Checkpoint loading
# =====================================================================

def load_checkpoint(
    checkpoint_path: Union[
        str,
        Path,
    ],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
) -> Tuple[
    int,
    Dict[str, float],
]:
    """
    Load a Stage-I checkpoint.

    Returns:

        start_epoch
        previous_losses
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    logger.info(
        f"Loading checkpoint: "
        f"{checkpoint_path}"
    )

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        # Compatibility with older PyTorch.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------
    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    # -------------------------------------------------------------
    # AMP scaler
    # -------------------------------------------------------------
    if (
        scaler is not None
        and "scaler_state_dict"
        in checkpoint
    ):

        scaler_state = checkpoint[
            "scaler_state_dict"
        ]

        if scaler_state:

            scaler.load_state_dict(
                scaler_state
            )

    completed_epoch = int(
        checkpoint.get(
            "epoch",
            0,
        )
    )

    losses = checkpoint.get(
        "losses",
        {},
    )

    start_epoch = (
        completed_epoch + 1
    )

    logger.info(
        "Checkpoint loaded successfully."
    )

    logger.info(
        f"Completed epoch: "
        f"{completed_epoch}"
    )

    logger.info(
        f"Resume epoch: "
        f"{start_epoch}"
    )

    return (
        start_epoch,
        losses,
    )


# =====================================================================
# Main training function
# =====================================================================

def train_stage1(
    model: nn.Module,
    dataloader: Iterable[
        Tuple[
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    config: Stage1Config,
    optimizer: Optional[
        torch.optim.Optimizer
    ] = None,
    criterion: Optional[
        nn.Module
    ] = None,
    device: Optional[
        Union[
            str,
            torch.device,
        ]
    ] = None,
    max_steps_per_epoch: Optional[
        int
    ] = None,
    checkpoint_dir: Optional[
        Union[
            str,
            Path,
        ]
    ] = None,
    save_every: Optional[
        int
    ] = None,
    resume: Optional[
        Union[
            str,
            Path,
        ]
    ] = None,
) -> Dict[str, float]:
    """
    Main Stage-I training function.
    """

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------
    target_device = resolve_device(
        device
        if device is not None
        else config.device
    )

    logger.info(
        f"Stage-I device: "
        f"{target_device}"
    )

    if target_device.type == "cuda":

        logger.info(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------
    set_seed(
        config.seed
    )

    # -------------------------------------------------------------
    # AMP
    # -------------------------------------------------------------
    use_amp = amp_enabled(
        target_device,
        config,
    )

    logger.info(
        f"AMP enabled: "
        f"{use_amp}"
    )

    # -------------------------------------------------------------
    # Move model
    # -------------------------------------------------------------
    model = model.to(
        target_device
    )

    # -------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------
    loss_module = (
        criterion
        if criterion is not None
        else Stage1Loss(
            config=config
        )
    )

    loss_module = loss_module.to(
        target_device
    )

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------
    opt = (
        optimizer
        if optimizer is not None
        else torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    )

    # -------------------------------------------------------------
    # Scaler
    # -------------------------------------------------------------
    if use_amp:

        scaler = create_grad_scaler(
            enabled=True
        )

    else:

        scaler = NullScaler()

    # -------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------
    start_epoch = 1

    if resume is not None:

        (
            start_epoch,
            previous_losses,
        ) = load_checkpoint(
            checkpoint_path=resume,
            model=model,
            optimizer=opt,
            device=target_device,
            scaler=scaler,
        )

        logger.info(
            f"Previous checkpoint losses: "
            f"{previous_losses}"
        )

    # -------------------------------------------------------------
    # Checkpoint configuration
    # -------------------------------------------------------------
    actual_checkpoint_dir = (
        checkpoint_dir
        if checkpoint_dir is not None
        else config.checkpoint_dir
    )

    actual_save_every = (
        save_every
        if save_every is not None
        else config.save_every
    )

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------
    final_losses: Dict[
        str,
        float,
    ] = {}

    for epoch in range(
        start_epoch,
        config.num_epochs + 1,
    ):

        if target_device.type == "cuda":

            torch.cuda.reset_peak_memory_stats()

        logger.info(
            "=" * 75
        )

        logger.info(
            f"Starting Stage-I Epoch "
            f"{epoch}/{config.num_epochs}"
        )

        logger.info(
            "=" * 75
        )

        epoch_losses = train_epoch(
            model=model,
            dataloader=dataloader,
            criterion=loss_module,
            optimizer=opt,
            epoch=epoch,
            device=target_device,
            scaler=scaler,
            use_amp=use_amp,
            log_interval=config.log_every,
            max_steps=max_steps_per_epoch,
            gradient_accumulation_steps=(
                config.gradient_accumulation_steps
            ),
            grad_clip_norm=(
                config.grad_clip_norm
            ),
            check_gradients_enabled=(
                config.check_gradients
            ),
        )

        final_losses = epoch_losses

        # ---------------------------------------------------------
        # Epoch summary
        # ---------------------------------------------------------
        logger.info(
            "-" * 75
        )

        logger.info(
            f"Epoch {epoch} Summary | "
            f"Total={epoch_losses.get('loss_total', 0.0):.6f} | "
            f"Struct={epoch_losses.get('loss_structure', 0.0):.6f} | "
            f"Decorr={epoch_losses.get('loss_decorr', 0.0):.6f} | "
            f"Comp={epoch_losses.get('loss_comp', 0.0):.6f} | "
            f"Info={epoch_losses.get('loss_info', 0.0):.6f} | "
            f"GradNorm={epoch_losses.get('grad_norm', 0.0):.4f} | "
            f"Time={epoch_losses.get('epoch_time', 0.0):.1f}s"
        )

        # ---------------------------------------------------------
        # CUDA memory
        # ---------------------------------------------------------
        log_cuda_memory(
            prefix="Epoch end | "
        )

        # ---------------------------------------------------------
        # Checkpoint
        # ---------------------------------------------------------
        if (
            config.save_checkpoints
            and actual_checkpoint_dir is not None
            and epoch % actual_save_every == 0
        ):

            save_checkpoint(
                model=model,
                optimizer=opt,
                epoch=epoch,
                losses=epoch_losses,
                checkpoint_dir=(
                    actual_checkpoint_dir
                ),
                config=config,
                scaler=scaler,
            )

        # ---------------------------------------------------------
        # Memory cleanup
        # ---------------------------------------------------------
        if (
            config.empty_cache_after_epoch
        ):

            gc.collect()

            if target_device.type == "cuda":

                torch.cuda.empty_cache()

    logger.info(
        "=" * 75
    )

    logger.info(
        "Stage-I training complete."
    )

    logger.info(
        "=" * 75
    )

    return final_losses


# =====================================================================
# Dry run
# =====================================================================

def run_dry_run(
    config: Optional[
        Stage1Config
    ] = None,
    dry_run_image_size: int = 128,
    dry_run_steps: int = 2,
) -> None:
    """
    Run a small synthetic training test.

    Default:

        batch = 1
        image = 128x128
        steps = 2

    This verifies:

        - model construction
        - tensor shapes
        - forward pass
        - four losses
        - backward pass
        - optimizer update
        - gradient finiteness
        - AMP path when CUDA is available
    """

    logger.info(
        "=" * 75
    )

    logger.info(
        "DAPFUSION STAGE-I DRY RUN"
    )

    logger.info(
        f"Image size: "
        f"{dry_run_image_size}x"
        f"{dry_run_image_size}"
    )

    logger.info(
        f"Steps: "
        f"{dry_run_steps}"
    )

    logger.info(
        "=" * 75
    )

    if dry_run_image_size <= 0:

        raise ValueError(
            "dry_run_image_size must be positive."
        )

    if dry_run_steps <= 0:

        raise ValueError(
            "dry_run_steps must be positive."
        )

    # -------------------------------------------------------------
    # Use a dedicated small configuration.
    #
    # Keep the full methodology:
    #
    #   4 scan directions
    #
    # but use one Mamba block for a lightweight pipeline sanity
    # check.
    # -------------------------------------------------------------
    cfg = (
        config
        if config is not None
        else Stage1Config(
            image_height=(
                dry_run_image_size
            ),
            image_width=(
                dry_run_image_size
            ),
            batch_size=1,
            num_epochs=1,
            num_mamba_blocks=1,
            num_scan_directions=4,
        )
    )

    # -------------------------------------------------------------
    # Ensure supplied config matches dry-run image size.
    # -------------------------------------------------------------
    if (
        cfg.image_height
        != dry_run_image_size
        or cfg.image_width
        != dry_run_image_size
    ):

        cfg = Stage1Config(
            **{
                **cfg.to_dict(),
                "image_height":
                    dry_run_image_size,
                "image_width":
                    dry_run_image_size,
            }
        )

    set_seed(
        cfg.seed
    )

    device = resolve_device(
        cfg.device
    )

    logger.info(
        f"Dry-run device: "
        f"{device}"
    )

    # -------------------------------------------------------------
    # Clear CUDA memory.
    # -------------------------------------------------------------
    gc.collect()

    if device.type == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        log_cuda_memory(
            prefix="Before model | "
        )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    model = Stage1Model(
        config=cfg
    ).to(device)

    criterion = Stage1Loss(
        config=cfg
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    use_amp = amp_enabled(
        device,
        cfg,
    )

    if use_amp:

        scaler = create_grad_scaler(
            enabled=True
        )

    else:

        scaler = NullScaler()

    # -------------------------------------------------------------
    # Synthetic data.
    # -------------------------------------------------------------
    rgb = torch.randn(
        1,
        cfg.rgb_in_channels,
        dry_run_image_size,
        dry_run_image_size,
    )

    thermal = torch.randn(
        1,
        cfg.thermal_in_channels,
        dry_run_image_size,
        dry_run_image_size,
    )

    # -------------------------------------------------------------
    # Run exactly the requested number of iterations.
    # -------------------------------------------------------------
    for iteration in range(
        dry_run_steps
    ):

        logger.info(
            f"Dry-run iteration "
            f"{iteration + 1}/"
            f"{dry_run_steps}"
        )

        logger.info(
            f"  RGB: "
            f"{tuple(rgb.shape)}"
        )

        logger.info(
            f"  Thermal: "
            f"{tuple(thermal.shape)}"
        )

        t0 = time.time()

        losses = train_step(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            rgb=rgb,
            thermal=thermal,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            gradient_accumulation_steps=1,
            accumulation_index=0,
            grad_clip_norm=(
                cfg.grad_clip_norm
            ),
            check_gradients_enabled=True,
        )

        elapsed = (
            time.time()
            - t0
        )

        logger.info(
            f"  Total loss: "
            f"{losses['loss_total']:.6f}"
        )

        logger.info(
            f"  Structure: "
            f"{losses['loss_structure']:.6f}"
        )

        logger.info(
            f"  Decorrelation: "
            f"{losses['loss_decorr']:.6f}"
        )

        logger.info(
            f"  Complementarity: "
            f"{losses['loss_comp']:.6f}"
        )

        logger.info(
            f"  Information: "
            f"{losses['loss_info']:.6f}"
        )

        logger.info(
            f"  Gradient norm: "
            f"{losses['grad_norm']:.6f}"
        )

        logger.info(
            f"  Time: "
            f"{elapsed:.3f}s"
        )

        if device.type == "cuda":

            log_cuda_memory(
                prefix="  "
            )

    # -------------------------------------------------------------
    # Final cleanup.
    # -------------------------------------------------------------
    del rgb
    del thermal
    del model
    del criterion
    del optimizer
    del scaler

    gc.collect()

    if device.type == "cuda":

        torch.cuda.empty_cache()

    logger.info(
        "=" * 75
    )

    logger.info(
        "STAGE-I DRY RUN SUCCEEDED."
    )

    logger.info(
        "=" * 75
    )


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    """
    Command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train DAPFusion Stage I "
            "Feature Representation Backbone."
        )
    )

    # -----------------------------------------------------------------
    # Dry run
    # -----------------------------------------------------------------

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run a short synthetic sanity test."
        ),
    )

    parser.add_argument(
        "--dry-run-size",
        type=int,
        default=128,
        help=(
            "Synthetic dry-run image size. "
            "Default: 128."
        ),
    )

    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=2,
        help=(
            "Number of synthetic dry-run steps. "
            "Default: 2."
        ),
    )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Root LLVIP directory. "
            "Expected subdirectories: "
            "visible/train, infrared/train, etc."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=[
            "train",
            "test",
        ],
        help=(
            "Dataset split. "
            "Default: train."
        ),
    )

    # -----------------------------------------------------------------
    # Image size
    # -----------------------------------------------------------------

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        choices=[
            128,
            192,
            256,
        ],
        help=(
            "Input image resolution. "
            "Supported: 128, 192, 256."
        ),
    )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Physical batch size. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help=(
            "Number of Stage-I training epochs."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help=(
            "AdamW learning rate."
        ),
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help=(
            "AdamW weight decay."
        ),
    )

    # -----------------------------------------------------------------
    # Gradient accumulation
    # -----------------------------------------------------------------

    parser.add_argument(
        "--grad-accumulation",
        type=int,
        default=1,
        help=(
            "Gradient accumulation steps. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help=(
            "Maximum gradient norm. "
            "Default: 1.0."
        ),
    )

    # -----------------------------------------------------------------
    # Mamba
    # -----------------------------------------------------------------

    parser.add_argument(
        "--mamba-backend",
        type=str,
        choices=[
            "custom_ssm",
            "mamba_ssm",
        ],
        default="custom_ssm",
        help=(
            "Mamba implementation."
        ),
    )

    parser.add_argument(
        "--scan-directions",
        type=int,
        choices=[
            1,
            4,
        ],
        default=4,
        help=(
            "Number of spatial scan directions."
        ),
    )

    parser.add_argument(
        "--mamba-blocks",
        type=int,
        default=4,
        help=(
            "Number of stacked Mamba blocks."
        ),
    )

    # -----------------------------------------------------------------
    # AMP
    # -----------------------------------------------------------------

    parser.add_argument(
        "--amp",
        action="store_true",
        default=None,
        help=(
            "Enable AMP."
        ),
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help=(
            "Disable AMP."
        ),
    )

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Device: auto, cuda, cuda:0, or cpu."
        ),
    )

    # -----------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker count."
        ),
    )

    # -----------------------------------------------------------------
    # Maximum steps
    # -----------------------------------------------------------------

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Maximum training steps per epoch. "
            "Useful for smoke tests."
        ),
    )

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Checkpoint output directory."
        ),
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help=(
            "Save checkpoint every N epochs."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Checkpoint path for resuming training."
        ),
    )

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help=(
            "Log every N batches."
        ),
    )

    # -----------------------------------------------------------------
    # Seed
    # -----------------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed."
        ),
    )

    args = parser.parse_args()

    # =================================================================
    # AMP selection
    # =================================================================

    if args.no_amp:

        use_amp = False

    elif args.amp is not None:

        use_amp = True

    else:

        # Default behavior:
        # enable AMP when CUDA is available.
        use_amp = torch.cuda.is_available()

    # =================================================================
    # Configuration
    # =================================================================

    config = Stage1Config(
        image_height=args.image_size,
        image_width=args.image_size,

        batch_size=args.batch_size,

        num_epochs=args.epochs,

        learning_rate=args.lr,

        weight_decay=args.weight_decay,

        mamba_backend=(
            args.mamba_backend
        ),

        num_scan_directions=(
            args.scan_directions
        ),

        num_mamba_blocks=(
            args.mamba_blocks
        ),

        use_amp=use_amp,

        gradient_accumulation_steps=(
            args.grad_accumulation
        ),

        grad_clip_norm=(
            args.grad_clip
        ),

        device=args.device,

        num_workers=args.num_workers,

        save_every=args.save_every,

        log_every=args.log_every,

        seed=args.seed,

        checkpoint_dir=(
            args.checkpoint_dir
            if args.checkpoint_dir is not None
            else "checkpoints/stage1"
        ),
    )

    # =================================================================
    # Dry run
    # =================================================================

    if args.dry_run:

        run_dry_run(
            config=config,
            dry_run_image_size=(
                args.dry_run_size
            ),
            dry_run_steps=(
                args.dry_run_steps
            ),
        )

        return

    # =================================================================
    # Real dataset
    # =================================================================

    if args.data_dir is None:

        logger.error(
            "No --data-dir was provided."
        )

        logger.info(
            "Use --dry-run for a synthetic "
            "pipeline test."
        )

        parser.print_help()

        return

    # -----------------------------------------------------------------
    # Import dataset factory only when real training is requested.
    # -----------------------------------------------------------------

    from dapfusion.stage1.dataset import (
        create_llvip_dataloader,
    )

    data_path = Path(
        args.data_dir
    )

    logger.info(
        f"LLVIP root: "
        f"{data_path}"
    )

    logger.info(
        f"Split: "
        f"{args.split}"
    )

    logger.info(
        f"Image size: "
        f"{config.image_height}x"
        f"{config.image_width}"
    )

    logger.info(
        f"Batch size: "
        f"{config.batch_size}"
    )

    logger.info(
        f"Gradient accumulation: "
        f"{config.gradient_accumulation_steps}"
    )

    # -----------------------------------------------------------------
    # Dataset / DataLoader
    # -----------------------------------------------------------------

    dataloader = create_llvip_dataloader(
        root_dir=data_path,
        split=args.split,
        batch_size=config.batch_size,
        shuffle=(
            args.split == "train"
        ),
        num_workers=config.num_workers,
        image_size=(
            config.image_height,
            config.image_width,
        ),
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    logger.info(
        f"Number of paired samples: "
        f"{len(dataloader.dataset)}"
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    model = Stage1Model(
        config=config
    )

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------

    train_stage1(
        model=model,
        dataloader=dataloader,
        config=config,
        device=args.device,
        max_steps_per_epoch=(
            args.max_steps
        ),
        checkpoint_dir=(
            args.checkpoint_dir
            if args.checkpoint_dir is not None
            else config.checkpoint_dir
        ),
        save_every=args.save_every,
        resume=args.resume,
    )


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":

    main()