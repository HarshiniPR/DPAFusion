"""
Stage 1 Training Pipeline Skeleton (Stage I, Section 3.1)
Reference: DAPFusion Framework - Stage I: Multi-Modal Feature Representation Backbone

Orchestrates Stage 1 training:
- Accepts any PyTorch DataLoader yielding (rgb, thermal) paired tensors.
- Supports AdamW optimizer (learning_rate=5e-5).
- Performs forward pass through Stage1Model, computes multi-objective Stage1Loss.
- Performs backpropagation and optimizer step.
- Logs all individual loss components per step.
- Provides --dry-run CLI mode running 2 iterations on synthetic tensors.
- Provides optional integration with PairedRGBThermalDataset / LLVIP dataset.
- Supports persistent checkpoint saving.
- Supports resuming training from a checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

# Ensure workspace root is in sys.path for direct script execution
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dapfusion.stage1.config import Stage1Config
from dapfusion.stage1.losses import Stage1Loss
from dapfusion.stage1.stage1_model import Stage1Model


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("TrainStage1")


# ---------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------

def train_step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Perform a single optimization step."""

    model.train()

    rgb = rgb.to(device)
    thermal = thermal.to(device)

    optimizer.zero_grad()

    model_outputs = model(rgb, thermal)

    losses = criterion(model_outputs)

    total_loss = losses["loss_total"]

    total_loss.backward()

    optimizer.step()

    return {
        k: v.item()
        for k, v in losses.items()
    }


# ---------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    log_interval: int = 10,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Train for a single epoch."""

    running_losses: Dict[str, float] = {}
    step_count = 0

    for step, (rgb, thermal) in enumerate(dataloader):

        if max_steps is not None and step >= max_steps:
            break

        step_t0 = time.time()

        step_losses = train_step(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            rgb=rgb,
            thermal=thermal,
            device=device,
        )

        step_time = time.time() - step_t0

        step_count += 1

        for k, v in step_losses.items():
            running_losses[k] = running_losses.get(k, 0.0) + v

        if (
            (step + 1) % log_interval == 0
            or (max_steps is not None and (step + 1) == max_steps)
        ):
            log_str = (
                f"Epoch [{epoch:02d}] Step [{step + 1:04d}] "
                f"Total: {step_losses['loss_total']:.4f} | "
                f"Struct: {step_losses['loss_structure']:.4f} | "
                f"Decorr: {step_losses['loss_decorr']:.4f} | "
                f"Comp: {step_losses['loss_comp']:.4f} | "
                f"Info: {step_losses['loss_info']:.4f} | "
                f"({step_time:.2f}s/it)"
            )

            logger.info(log_str)

    return {
        k: v / max(1, step_count)
        for k, v in running_losses.items()
    }


# ---------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    losses: Dict[str, float],
    checkpoint_dir: Union[str, Path],
    config: Stage1Config,
) -> Path:
    """
    Save a complete Stage 1 training checkpoint.

    The checkpoint contains:
        - current epoch
        - model state
        - optimizer state
        - epoch losses
        - configuration
    """

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "config": vars(config),
    }

    # Epoch-specific checkpoint
    epoch_path = checkpoint_dir / f"stage1_epoch_{epoch:03d}.pth"

    torch.save(
        checkpoint,
        epoch_path,
    )

    # Always update latest checkpoint
    latest_path = checkpoint_dir / "latest.pth"

    torch.save(
        checkpoint,
        latest_path,
    )

    logger.info(
        f"Checkpoint saved: {epoch_path}"
    )

    logger.info(
        f"Latest checkpoint updated: {latest_path}"
    )

    return latest_path


# ---------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, Dict[str, float]]:
    """
    Load a Stage 1 checkpoint.

    Returns:
        start_epoch:
            Epoch from which training should continue.

        losses:
            Loss dictionary stored in the checkpoint.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    logger.info(
        f"Loading checkpoint from: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    completed_epoch = checkpoint.get(
        "epoch",
        0,
    )

    losses = checkpoint.get(
        "losses",
        {},
    )

    start_epoch = completed_epoch + 1

    logger.info(
        f"Checkpoint loaded successfully."
    )

    logger.info(
        f"Previously completed epoch: {completed_epoch}"
    )

    logger.info(
        f"Training will resume from epoch: {start_epoch}"
    )

    return start_epoch, losses


# ---------------------------------------------------------------------
# Main Stage 1 training
# ---------------------------------------------------------------------

def train_stage1(
    model: nn.Module,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    config: Stage1Config,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
    max_steps_per_epoch: Optional[int] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    save_every: int = 1,
    resume: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Main training execution function for Stage 1.

    Args:
        model:
            Stage1Model.

        dataloader:
            RGB-Thermal training DataLoader.

        config:
            Stage1Config.

        optimizer:
            Optional optimizer. Defaults to AdamW.

        criterion:
            Optional Stage1Loss.

        device:
            Training device.

        max_steps_per_epoch:
            Optional limit on training steps per epoch.

        checkpoint_dir:
            Directory where checkpoints will be stored.

        save_every:
            Save checkpoint every N epochs.

        resume:
            Optional checkpoint path from which to resume training.
    """

    target_device = (
        torch.device(device)
        if device is not None
        else torch.device(
            config.device
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    logger.info(
        f"Initiating Stage 1 training on device: {target_device}"
    )

    if target_device.type == "cuda":
        logger.info(
            f"GPU: {torch.cuda.get_device_name(target_device)}"
        )

    # -------------------------------------------------------------
    # Move model and loss to device
    # -------------------------------------------------------------

    model = model.to(target_device)

    loss_module = criterion or Stage1Loss(
        config=config
    )

    loss_module = loss_module.to(
        target_device
    )

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------

    opt = optimizer or torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # -------------------------------------------------------------
    # Resume checkpoint
    # -------------------------------------------------------------

    start_epoch = 1

    if resume is not None:

        start_epoch, _ = load_checkpoint(
            checkpoint_path=resume,
            model=model,
            optimizer=opt,
            device=target_device,
        )

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    final_losses: Dict[str, float] = {}

    for epoch in range(
        start_epoch,
        config.num_epochs + 1,
    ):

        epoch_t0 = time.time()

        logger.info(
            f"--- Starting Epoch "
            f"{epoch}/{config.num_epochs} ---"
        )

        epoch_losses = train_epoch(
            model=model,
            dataloader=dataloader,
            criterion=loss_module,
            optimizer=opt,
            epoch=epoch,
            device=target_device,
            log_interval=1,
            max_steps=max_steps_per_epoch,
        )

        elapsed = time.time() - epoch_t0

        logger.info(
            f"Epoch {epoch} complete in "
            f"{elapsed:.1f}s. "
            f"Avg Total Loss: "
            f"{epoch_losses.get('loss_total', 0.0):.4f}"
        )

        final_losses = epoch_losses

        # ---------------------------------------------------------
        # Checkpoint saving
        # ---------------------------------------------------------

        if (
            checkpoint_dir is not None
            and epoch % save_every == 0
        ):

            save_checkpoint(
                model=model,
                optimizer=opt,
                epoch=epoch,
                losses=epoch_losses,
                checkpoint_dir=checkpoint_dir,
                config=config,
            )

    return final_losses


# ---------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------
def run_dry_run(
    config: Optional[Stage1Config] = None,
) -> None:

    import gc

    cfg = config or Stage1Config(
        batch_size=1,
        num_epochs=1,
    )

    logger.info("=" * 65)

    logger.info(
        "RUNNING DAPFUSION STAGE 1 "
        "SANITY DRY-RUN (2 ITERATIONS)"
    )

    logger.info("=" * 65)

    # ---------------------------------------------------------
    # Clean up any memory left by previous CUDA operations
    # ---------------------------------------------------------
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(
        cfg.device
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        f"Dry-run device: {device}"
    )

    if torch.cuda.is_available():
        logger.info(
            f"GPU memory allocated before model: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------
    model = Stage1Model(
        config=cfg
    ).to(device)

    criterion = Stage1Loss(
        config=cfg
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
    )

    # ---------------------------------------------------------
    # Lightweight synthetic data
    #
    # IMPORTANT:
    # batch_size = 1
    # resolution = 128 x 128
    #
    # This is intentionally smaller than the actual
    # 256 x 256 training resolution to prevent OOM during
    # architecture / gradient sanity checking.
    # ---------------------------------------------------------
    dry_run_batch_size = 1
    dry_run_image_size = 128

    synthetic_rgb = torch.randn(
        dry_run_batch_size,
        3,
        dry_run_image_size,
        dry_run_image_size,
    )

    synthetic_thermal = torch.randn(
        dry_run_batch_size,
        1,
        dry_run_image_size,
        dry_run_image_size,
    )

    synthetic_dataset = TensorDataset(
        synthetic_rgb,
        synthetic_thermal,
    )

    synthetic_loader = DataLoader(
        synthetic_dataset,
        batch_size=dry_run_batch_size,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Run exactly 2 sanity-check iterations
    # ---------------------------------------------------------
    for i, (
        rgb_batch,
        thermal_batch,
    ) in enumerate(synthetic_loader):

        if i >= 2:
            break

        logger.info(
            f"[Dry-Run Iteration {i + 1}/2]"
        )

        logger.info(
            f"  Input RGB shape:     "
            f"{tuple(rgb_batch.shape)}"
        )

        logger.info(
            f"  Input Thermal shape: "
            f"{tuple(thermal_batch.shape)}"
        )

        step_t0 = time.time()

        step_losses = train_step(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            rgb=rgb_batch,
            thermal=thermal_batch,
            device=device,
        )

        step_time = (
            time.time() - step_t0
        )

        logger.info(
            f"  Step completed in "
            f"{step_time:.3f}s with losses:"
        )

        for loss_name, loss_val in step_losses.items():

            logger.info(
                f"    - "
                f"{loss_name:15s}: "
                f"{loss_val:.6f}"
            )

        # -----------------------------------------------------
        # Report GPU memory after each iteration
        # -----------------------------------------------------
        if torch.cuda.is_available():

            allocated_gb = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            reserved_gb = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

            logger.info(
                f"  GPU memory: "
                f"{allocated_gb:.2f} GB allocated / "
                f"{reserved_gb:.2f} GB reserved"
            )

            # Release temporary cached blocks
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Final cleanup
    # ---------------------------------------------------------
    del synthetic_rgb
    del synthetic_thermal
    del synthetic_dataset
    del synthetic_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("=" * 65)

    logger.info(
        "SANITY DRY-RUN SUCCEEDED! "
        "ALL TENSOR SHAPES & GRADIENTS VERIFIED."
    )

    logger.info("=" * 65)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint for Stage 1 training."""

    parser = argparse.ArgumentParser(
        description=(
            "Train DAPFusion Stage 1: "
            "Feature Representation Backbone"
        )
    )

    # -------------------------------------------------------------
    # Dry run
    # -------------------------------------------------------------

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run 2 iterations on synthetic random "
            "tensors to verify pipeline shapes "
            "and gradients."
        ),
    )

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Optional path to dataset directory "
            "(e.g., path to LLVIP)."
        ),
    )

    # -------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for Adam/AdamW.",
    )

    # -------------------------------------------------------------
    # Mamba configuration
    # -------------------------------------------------------------

    parser.add_argument(
        "--mamba-backend",
        type=str,
        choices=[
            "mamba_ssm",
            "custom_ssm",
        ],
        default="custom_ssm",
        help=(
            "Mamba backend "
            "('mamba_ssm' or 'custom_ssm')."
        ),
    )

    parser.add_argument(
        "--scan-directions",
        type=int,
        choices=[1, 4],
        default=4,
        help=(
            "Number of scan directions "
            "(1 or 4)."
        ),
    )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        help=(
            "Execution device "
            "('cuda' or 'cpu')."
        ),
    )

    # -------------------------------------------------------------
    # Maximum steps
    # -------------------------------------------------------------

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional cap on training "
            "steps per epoch."
        ),
    )

    # -------------------------------------------------------------
    # Checkpoint directory
    # -------------------------------------------------------------

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory where Stage 1 "
            "checkpoints will be saved."
        ),
    )

    # -------------------------------------------------------------
    # Checkpoint frequency
    # -------------------------------------------------------------

    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help=(
            "Save a checkpoint every N epochs."
        ),
    )

    # -------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint to resume "
            "training from."
        ),
    )

    # Parse arguments

    args = parser.parse_args()

    # Build configuration
    config = Stage1Config(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        mamba_backend=args.mamba_backend,
        num_scan_directions=args.scan_directions,
        device=args.device,
    )

    # Dry run  
    if args.dry_run:

        run_dry_run(config)

        return

    # -------------------------------------------------------------
    # Real dataset training
    # -------------------------------------------------------------

    if args.data_dir is not None:

        from dapfusion.stage1.dataset import (
            create_llvip_dataloader,
        )

        data_path = Path(
            args.data_dir
        )

        logger.info(
            f"Loading dataset from: "
            f"{data_path}"
        )

        dataloader = create_llvip_dataloader(
            root_dir=data_path,
            split="train",
            batch_size=config.batch_size,
            shuffle=True,
        )

        model = Stage1Model(
            config=config
        )

        train_stage1(
            model=model,
            dataloader=dataloader,
            config=config,
            device=args.device,
            max_steps_per_epoch=args.max_steps,
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
            resume=args.resume,
        )

    else:

        logger.warning(
            "No --data-dir provided and "
            "--dry-run not set. "
            "Please provide a DataLoader "
            "or use --dry-run to test."
        )

        parser.print_help()

if __name__ == "__main__":
    main()