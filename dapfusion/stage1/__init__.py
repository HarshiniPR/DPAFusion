"""
DAPFusion Stage 1
=================

Feature Representation Backbone for RGB-Thermal Image Fusion.
"""

# ---------------------------------------------------------------------
# CNN feature extraction
# ---------------------------------------------------------------------

from dapfusion.stage1.cnn_stem import (
    CNNStem,
)


# ---------------------------------------------------------------------
# Complementarity estimation
# ---------------------------------------------------------------------

from dapfusion.stage1.complementarity import (
    ComplementarityEstimationModule,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

from dapfusion.stage1.config import (
    Stage1Config,
)


# ---------------------------------------------------------------------
# Cross-modal interaction
# ---------------------------------------------------------------------

from dapfusion.stage1.cross_interaction import (
    CrossModalInteraction,
)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

from dapfusion.stage1.dataset import (
    PairedRGBThermalDataset,
    create_llvip_dataloader,
)


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------

from dapfusion.stage1.losses import (
    Stage1Loss,
    complementarity_diversity_loss,
    feature_decorrelation_loss,
    feature_information_preservation_loss,
    normalize_features,
    sobel_gradient_magnitude,
    structure_preservation_loss,
)


# ---------------------------------------------------------------------
# Mamba / selective SSM
# ---------------------------------------------------------------------

from dapfusion.stage1.mamba_encoder import (
    CustomSelectiveScanSSM,
    FastSelectiveScanRecurrence,
    VisionMambaBlock,
    VisionMambaEncoder,
)


# ---------------------------------------------------------------------
# Complete Stage 1 model
# ---------------------------------------------------------------------

from dapfusion.stage1.stage1_model import (
    Stage1Model,
    UnifiedFeatureFusion,
)


# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------

from dapfusion.stage1.train_stage1 import (
    run_dry_run,
    train_epoch,
    train_stage1,
    train_step,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [

    # -------------------------------------------------------------
    # CNN
    # -------------------------------------------------------------

    "CNNStem",

    # -------------------------------------------------------------
    # Mamba / SSM
    # -------------------------------------------------------------

    "VisionMambaBlock",
    "VisionMambaEncoder",
    "CustomSelectiveScanSSM",
    "FastSelectiveScanRecurrence",

    # -------------------------------------------------------------
    # Complementarity
    # -------------------------------------------------------------

    "ComplementarityEstimationModule",

    # -------------------------------------------------------------
    # Cross-modal interaction
    # -------------------------------------------------------------

    "CrossModalInteraction",

    # -------------------------------------------------------------
    # Fusion
    # -------------------------------------------------------------

    "UnifiedFeatureFusion",
    "Stage1Model",

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    "Stage1Config",

    # -------------------------------------------------------------
    # Losses
    # -------------------------------------------------------------

    "Stage1Loss",
    "sobel_gradient_magnitude",
    "structure_preservation_loss",
    "feature_decorrelation_loss",
    "complementarity_diversity_loss",
    "feature_information_preservation_loss",
    "normalize_features",

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    "PairedRGBThermalDataset",
    "create_llvip_dataloader",

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    "train_step",
    "train_epoch",
    "train_stage1",
    "run_dry_run",
]