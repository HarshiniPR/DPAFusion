"""
Stage I Configuration Module
============================

Reference:
    DAPFusion Framework - Stage I:
    Multi-Modal Feature Representation Backbone

Defines all hyperparameters for:

1. Inputs
2. Shallow CNN Feature Extraction (Section 3.1.1)
3. Dual Mamba Context Encoder (Section 3.1.2)
4. Complementarity Estimation Module (Section 3.1.3)
5. Complementarity-Guided Cross-Modal Interaction (Section 3.1.4)
6. Unified Feature Representation (Section 3.1.5)
7. Loss Objectives (Section 3.1.6)
8. Optimization and Training
9. Runtime / Memory Management
10. Checkpointing
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Union


@dataclass
class Stage1Config:
    """
    Configuration dataclass for DAPFusion Stage I.

    The architecture itself is kept fixed across image-size ablations.
    Only the input spatial resolution changes.

    Supported development / ablation sizes:

        128 x 128
        192 x 192
        256 x 256

    The final Stage-I experiment can use 256 x 256.
    """

    # =================================================================
    # 1. INPUTS
    # =================================================================

    rgb_in_channels: int = 3

    thermal_in_channels: int = 1

    # -------------------------------------------------------------
    # Image size
    #
    # Keep these configurable because image-size ablation is part
    # of the experimental protocol.
    # -------------------------------------------------------------
    image_height: int = 256
    image_width: int = 256

    # Supported sizes are not enforced as a hard architectural
    # restriction. This allows future experiments with other
    # resolutions as well.
    #
    # Expected experiments:
    #   128 x 128
    #   192 x 192
    #   256 x 256

    # =================================================================
    # 2. SHALLOW CNN FEATURE EXTRACTION
    #    Section 3.1.1
    # =================================================================

    # Three CNN stages:
    #
    #   C_in -> 32 -> 64 -> 128
    #
    # with stride pattern:
    #
    #   1 -> 2 -> 2
    #
    # resulting in 4x spatial downsampling.
    stem_channels: Tuple[int, int, int] = (32, 64, 128)

    stem_kernel_size: int = 3

    stem_padding: int = 1

    stem_leaky_relu_slope: float = 0.1

    # =================================================================
    # 3. DUAL MAMBA CONTEXT ENCODER
    #    Section 3.1.2
    # =================================================================

    d_model: int = 128

    d_state: int = 16

    expand_ratio: int = 2

    mamba_dropout: float = 0.1

    num_mamba_blocks: int = 4

    # Backend:
    #
    #   "mamba_ssm"
    #       Use the official Mamba implementation when available.
    #
    #   "custom_ssm"
    #       Use the project custom selective state-space
    #       implementation.
    #
    # custom_ssm remains the default because it is the fallback
    # implementation used by the project.
    mamba_backend: Literal[
        "mamba_ssm",
        "custom_ssm",
    ] = "custom_ssm"

    # Number of spatial scan directions.
    #
    # 1:
    #   lightweight development/debugging mode
    #
    # 4:
    #   full bidirectional / multi-directional spatial context
    #   used by the intended methodology.
    num_scan_directions: int = 4

    scan_merge: Literal[
        "average",
        "concat",
    ] = "average"

    # =================================================================
    # 4. COMPLEMENTARITY ESTIMATION MODULE
    #    Section 3.1.3
    # =================================================================

    # CEM descriptor:
    #
    #   [zr, zt, |zr - zt|, zr * zt]
    #
    # For C=128:
    #
    #   4 * 128 = 512
    cem_in_dim: int = 512

    cem_hidden_dim: int = 256

    cem_out_dim: int = 128

    # =================================================================
    # 5. CROSS-MODAL INTERACTION
    #    Section 3.1.4
    # =================================================================

    # Initial learnable cross-modal interaction coefficient.
    init_alpha: float = 1.0

    # =================================================================
    # 6. UNIFIED FEATURE REPRESENTATION
    #    Section 3.1.5
    # =================================================================

    # RGB and Thermal refined features are concatenated:
    #
    #   128 + 128 = 256
    fusion_in_channels: int = 256

    fusion_out_channels: int = 128

    fusion_activation: Literal[
        "bn_leaky_relu",
        "none",
    ] = "bn_leaky_relu"

    fusion_leaky_relu_slope: float = 0.1

    # =================================================================
    # 7. LOSS OBJECTIVES
    #    Section 3.1.6
    # =================================================================

    # -------------------------------------------------------------
    # Overall Stage-I objective:
    #
    # L =
    #     lambda_structure * L_structure
    #   + lambda_decorr    * L_decorr
    #   + lambda_comp      * L_comp
    #   + lambda_info      * L_info
    # -------------------------------------------------------------

    lambda_structure: float = 1.0

    lambda_decorr: float = 0.5

    lambda_comp: float = 0.1

    lambda_info: float = 1.0

    # -------------------------------------------------------------
    # Feature decorrelation
    # -------------------------------------------------------------
    decorr_tau: float = 0.3

    # -------------------------------------------------------------
    # Complementarity regularization
    # -------------------------------------------------------------
    comp_target_mean: float = 0.5

    comp_min_std: float = 0.05

    # -------------------------------------------------------------
    # Information preservation reference
    # -------------------------------------------------------------
    info_loss_reference: Literal[
        "context",
        "original_cnn_output",
    ] = "context"

    # -------------------------------------------------------------
    # Feature normalization
    # -------------------------------------------------------------
    normalize_before_loss: bool = True

    feature_norm_type: Literal[
        "l2",
        "instance",
    ] = "l2"

    # =================================================================
    # 8. OPTIMIZATION & TRAINING
    # =================================================================

    # Learning rate from the current Stage-I configuration.
    learning_rate: float = 5e-5

    weight_decay: float = 1e-4

    # -------------------------------------------------------------
    # Batch size
    #
    # Default is intentionally conservative for the ~16 GB GPU
    # environment because the Mamba encoder can have a large
    # activation footprint.
    # -------------------------------------------------------------
    batch_size: int = 1

    # Number of epochs for a normal Stage-I run.
    num_epochs: int = 10

    # Device:
    #
    #   "cpu"
    #   "cuda"
    #   "auto"
    #
    # "auto" selects CUDA when available.
    device: str = "auto"

    # -------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------
    seed: int = 42

    # -------------------------------------------------------------
    # Gradient clipping
    #
    # Important for stable optimization of the custom SSM/Mamba
    # component.
    # -------------------------------------------------------------
    grad_clip_norm: float = 1.0

    # =================================================================
    # 9. MIXED PRECISION / MEMORY MANAGEMENT
    # =================================================================

    # Enable AMP when CUDA is available.
    #
    # IMPORTANT:
    # The custom SSM implementation is responsible for keeping
    # numerically sensitive recurrence operations in FP32.
    # AMP here is a training-level optimization only.
    use_amp: bool = True

    # Gradient accumulation allows an effective batch size larger
    # than the physical GPU batch size.
    #
    # Example:
    #
    #   batch_size = 1
    #   gradient_accumulation_steps = 4
    #
    # gives an effective batch size of approximately 4.
    gradient_accumulation_steps: int = 1

    # Free cached CUDA memory after an epoch.
    empty_cache_after_epoch: bool = True

    # Number of DataLoader workers.
    #
    # Use 0 by default for maximum compatibility with Windows and
    # Colab notebook execution.
    num_workers: int = 0

    # Pin CPU memory when CUDA is used.
    pin_memory: bool = True

    # Persistent workers only make sense when num_workers > 0.
    persistent_workers: bool = False

    # =================================================================
    # 10. CHECKPOINTING
    # =================================================================

    # Enable checkpoint saving.
    save_checkpoints: bool = True

    # Save a checkpoint every N epochs.
    save_every: int = 1

    # Directory for Stage-I checkpoints.
    checkpoint_dir: str = "checkpoints/stage1"

    # Automatically resume from the latest checkpoint when one exists.
    resume: bool = False

    # =================================================================
    # 11. LOGGING / DIAGNOSTICS
    # =================================================================

    # Print individual loss components.
    log_loss_components: bool = True

    # Print gradient norm.
    log_gradient_norm: bool = True

    # Check model gradients for NaN / Inf.
    check_gradients: bool = True

    # Frequency of training progress messages.
    log_every: int = 10

    # =================================================================
    # VALIDATION
    # =================================================================

    # Whether to run a validation pass if a validation loader is
    # supplied by the training script.
    run_validation: bool = True

    # =================================================================
    # VALIDATION HELPERS
    # =================================================================

    def __post_init__(self) -> None:
        """
        Validate configuration values after initialization.

        These checks validate configuration consistency but do not
        impose a fixed image resolution.
        """

        if self.rgb_in_channels <= 0:
            raise ValueError(
                "rgb_in_channels must be positive."
            )

        if self.thermal_in_channels <= 0:
            raise ValueError(
                "thermal_in_channels must be positive."
            )

        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError(
                "image_height and image_width must be positive."
            )

        if len(self.stem_channels) != 3:
            raise ValueError(
                "stem_channels must contain exactly three values."
            )

        if any(c <= 0 for c in self.stem_channels):
            raise ValueError(
                "All stem channel sizes must be positive."
            )

        if self.d_model <= 0:
            raise ValueError(
                "d_model must be positive."
            )

        if self.d_state <= 0:
            raise ValueError(
                "d_state must be positive."
            )

        if self.expand_ratio <= 0:
            raise ValueError(
                "expand_ratio must be positive."
            )

        if self.num_mamba_blocks <= 0:
            raise ValueError(
                "num_mamba_blocks must be positive."
            )

        if self.num_scan_directions not in (1, 4):
            raise ValueError(
                "num_scan_directions must be either 1 or 4."
            )

        if self.cem_in_dim <= 0:
            raise ValueError(
                "cem_in_dim must be positive."
            )

        if self.cem_hidden_dim <= 0:
            raise ValueError(
                "cem_hidden_dim must be positive."
            )

        if self.cem_out_dim <= 0:
            raise ValueError(
                "cem_out_dim must be positive."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.num_epochs <= 0:
            raise ValueError(
                "num_epochs must be positive."
            )

        if self.grad_clip_norm <= 0:
            raise ValueError(
                "grad_clip_norm must be positive."
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        if self.save_every <= 0:
            raise ValueError(
                "save_every must be positive."
            )

        if self.log_every <= 0:
            raise ValueError(
                "log_every must be positive."
            )

    # =================================================================
    # SERIALIZATION
    # =================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Tuples are converted to lists automatically by JSON
        serialization when necessary.
        """
        data = asdict(self)

        # Keep JSON/YAML representation simple and portable.
        data["stem_channels"] = list(self.stem_channels)

        return data

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "Stage1Config":
        """
        Instantiate configuration from a dictionary.

        Unknown keys are ignored so that configuration files from
        earlier versions remain usable.
        """

        if not isinstance(data, dict):
            raise TypeError(
                "Stage1Config.from_dict expects a dictionary."
            )

        valid_keys = cls.__dataclass_fields__.keys()

        filtered_data = {
            key: value
            for key, value in data.items()
            if key in valid_keys
        }

        if "stem_channels" in filtered_data:
            channels = filtered_data["stem_channels"]

            if isinstance(channels, list):
                filtered_data["stem_channels"] = tuple(channels)

            elif isinstance(channels, tuple):
                filtered_data["stem_channels"] = channels

        return cls(**filtered_data)

    # =================================================================
    # JSON
    # =================================================================

    def save_json(
        self,
        file_path: Union[str, Path],
    ) -> None:
        """
        Save configuration to a JSON file.
        """

        path = Path(file_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.to_dict(),
                f,
                indent=4,
            )

    @classmethod
    def load_json(
        cls,
        file_path: Union[str, Path],
    ) -> "Stage1Config":
        """
        Load configuration from a JSON file.
        """

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return cls.from_dict(data)

    # =================================================================
    # YAML
    # =================================================================

    def save_yaml(
        self,
        file_path: Union[str, Path],
    ) -> None:
        """
        Save configuration to YAML if PyYAML is installed.

        If PyYAML is unavailable, save JSON with the same base name.
        """

        try:
            import yaml  # type: ignore

        except ImportError:
            self.save_json(
                Path(file_path).with_suffix(".json")
            )
            return

        path = Path(file_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                sort_keys=False,
            )

    @classmethod
    def load_yaml(
        cls,
        file_path: Union[str, Path],
    ) -> "Stage1Config":
        """
        Load configuration from YAML if PyYAML is installed.

        If PyYAML is unavailable, attempt to load the corresponding
        JSON file.
        """

        try:
            import yaml  # type: ignore

        except ImportError:
            return cls.load_json(
                Path(file_path).with_suffix(".json")
            )

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls.from_dict(data)


# =====================================================================
# Standalone verification
# =====================================================================

if __name__ == "__main__":

    print("Testing Stage1Config...")

    # -------------------------------------------------------------
    # Default configuration
    # -------------------------------------------------------------
    config = Stage1Config()

    print("\nDefault configuration:")
    print(f"Image size: {config.image_height}x{config.image_width}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Mamba backend: {config.mamba_backend}")
    print(f"Scan directions: {config.num_scan_directions}")
    print(f"AMP: {config.use_amp}")

    # -------------------------------------------------------------
    # Image-size configurations
    # -------------------------------------------------------------
    for size in (128, 192, 256):

        test_config = Stage1Config(
            image_height=size,
            image_width=size,
        )

        assert test_config.image_height == size
        assert test_config.image_width == size

        print(
            f"Image-size configuration verified: "
            f"{size}x{size}"
        )

    # -------------------------------------------------------------
    # Dictionary round trip
    # -------------------------------------------------------------
    data = config.to_dict()

    restored = Stage1Config.from_dict(data)

    assert restored.image_height == config.image_height
    assert restored.image_width == config.image_width
    assert restored.stem_channels == config.stem_channels

    print("Dictionary serialization verified.")

    print("\nStage1Config verified successfully!")