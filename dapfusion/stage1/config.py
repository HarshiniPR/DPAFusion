"""
Stage I Configuration Module
Reference: DAPFusion Framework - Stage I: Multi-Modal Feature Representation Backbone

Defines all hyperparameters for:
- Shallow CNN Feature Extraction (Section 3.1.1)
- Dual Mamba Context Encoder (Section 3.1.2)
- Complementarity Estimation Module (Section 3.1.3)
- Complementarity-Guided Cross-Modal Interaction (Section 3.1.4)
- Unified Feature Representation (Section 3.1.5)
- Loss Objectives (Section 3.1.6)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Union


@dataclass
class Stage1Config:
    """Configuration dataclass holding all hyperparameters for DAPFusion Stage 1."""

    # 1. Inputs
    rgb_in_channels: int = 3
    thermal_in_channels: int = 1
    image_height: int = 256
    image_width: int = 256

    # 2. Shallow CNN Feature Extraction (Section 3.1.1)
    stem_channels: Tuple[int, int, int] = (32, 64, 128)
    stem_kernel_size: int = 3
    stem_padding: int = 1
    stem_leaky_relu_slope: float = 0.1

    # 3. Dual Mamba Context Encoder (Section 3.1.2)
    d_model: int = 128
    d_state: int = 16
    expand_ratio: int = 2
    mamba_dropout: float = 0.1
    num_mamba_blocks: int = 4
    mamba_backend: Literal["mamba_ssm", "custom_ssm"] = "custom_ssm"
    num_scan_directions: int = 4  # 1 or 4
    scan_merge: Literal["average", "concat"] = "average"

    # 4. Complementarity Estimation Module (Section 3.1.3)
    cem_in_dim: int = 512  # [zr, zt, |zr - zt|, zr * zt] = 128 * 4
    cem_hidden_dim: int = 256
    cem_out_dim: int = 128

    # 5. Cross-Modal Interaction (Section 3.1.4)
    init_alpha: float = 1.0

    # 6. Unified Feature Representation (Section 3.1.5)
    fusion_in_channels: int = 256  # Fr_prime (128) + Ft_prime (128)
    fusion_out_channels: int = 128
    fusion_activation: Literal["bn_leaky_relu", "none"] = "bn_leaky_relu"
    fusion_leaky_relu_slope: float = 0.1

    # 7. Loss Objectives & Hyperparameters (Section 3.1.6)
    lambda_structure: float = 1.0
    lambda_decorr: float = 0.5
    lambda_comp: float = 0.1
    lambda_info: float = 1.0

    decorr_tau: float = 0.3
    comp_target_mean: float = 0.5
    comp_min_std: float = 0.05
    info_loss_reference: Literal["context", "original_cnn_output"] = "context"
    normalize_before_loss: bool = True
    feature_norm_type: Literal["l2", "instance"] = "l2"

    # 8. Optimization & Training Settings
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 10
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stage1Config:
        """Instantiate configuration from dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        if "stem_channels" in filtered_data and isinstance(filtered_data["stem_channels"], list):
            filtered_data["stem_channels"] = tuple(filtered_data["stem_channels"])
        return cls(**filtered_data)

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> Stage1Config:
        """Load configuration from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file if PyYAML is available, else fallback to JSON."""
        try:
            import yaml  # type: ignore
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except ImportError:
            self.save_json(Path(file_path).with_suffix(".json"))

    @classmethod
    def load_yaml(cls, file_path: Union[str, Path]) -> Stage1Config:
        """Load configuration from YAML file if PyYAML is available, else fallback to JSON."""
        try:
            import yaml  # type: ignore
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except ImportError:
            return cls.load_json(Path(file_path).with_suffix(".json"))
