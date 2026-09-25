"""
DAPFusion Stage 1: Feature Representation Backbone for RGB-Thermal Image Fusion.
"""

from dapfusion.stage1.cnn_stem import CNNStem
from dapfusion.stage1.complementarity import ComplementarityEstimationModule
from dapfusion.stage1.config import Stage1Config
from dapfusion.stage1.cross_interaction import CrossModalInteraction
from dapfusion.stage1.dataset import PairedRGBThermalDataset, create_llvip_dataloader
from dapfusion.stage1.losses import (
    Stage1Loss,
    complementarity_diversity_loss,
    feature_decorrelation_loss,
    feature_information_preservation_loss,
    normalize_features,
    sobel_gradient_magnitude,
    structure_preservation_loss,
)
from dapfusion.stage1.mamba_encoder import (
    CustomSelectiveScanSSM,
    FastSelectiveScanRecurrence,
    VisionMambaBlock,
    VisionMambaEncoder,
)
from dapfusion.stage1.stage1_model import Stage1Model, UnifiedFeatureFusion
from dapfusion.stage1.train_stage1 import run_dry_run, train_epoch, train_stage1, train_step

__all__ = [
    "CNNStem",
    "VisionMambaBlock",
    "VisionMambaEncoder",
    "CustomSelectiveScanSSM",
    "FastSelectiveScanRecurrence",
    "ComplementarityEstimationModule",
    "CrossModalInteraction",
    "UnifiedFeatureFusion",
    "Stage1Model",
    "Stage1Config",
    "Stage1Loss",
    "sobel_gradient_magnitude",
    "structure_preservation_loss",
    "feature_decorrelation_loss",
    "complementarity_diversity_loss",
    "feature_information_preservation_loss",
    "normalize_features",
    "train_step",
    "train_epoch",
    "train_stage1",
    "run_dry_run",
    "PairedRGBThermalDataset",
    "create_llvip_dataloader",
]
