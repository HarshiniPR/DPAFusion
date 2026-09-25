"""
Tests for DAPFusion Stage 1
===========================

These tests verify:

    1. Configuration
    2. CNN stem
    3. Complementarity module
    4. Cross-modal interaction
    5. Mamba encoder
    6. Complete Stage-I model
    7. Four Stage-I losses
    8. Forward + backward pass
    9. Multiple input resolutions
   10. Finite gradients

The tests intentionally use small tensors so they can run on CPU
without requiring the LLVIP dataset or a GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from dapfusion.stage1.cnn_stem import (
    CNNStem,
)

from dapfusion.stage1.complementarity import (
    ComplementarityEstimationModule,
)

from dapfusion.stage1.config import (
    Stage1Config,
)

from dapfusion.stage1.cross_interaction import (
    CrossModalInteraction,
)

from dapfusion.stage1.losses import (
    Stage1Loss,
)

from dapfusion.stage1.mamba_encoder import (
    VisionMambaEncoder,
)

from dapfusion.stage1.stage1_model import (
    Stage1Model,
)


# =====================================================================
# Test configuration
# =====================================================================

def make_test_config(
    image_size: int = 128,
) -> Stage1Config:
    """
    Create a lightweight configuration for unit tests.

    The architecture is unchanged.

    Only the number of Mamba blocks is reduced so tests remain
    practical on CPU.
    """

    return Stage1Config(
        image_height=image_size,
        image_width=image_size,

        batch_size=1,

        num_epochs=1,

        num_mamba_blocks=1,

        num_scan_directions=4,

        use_amp=False,

        device="cpu",

        check_gradients=True,

        save_checkpoints=False,
    )


# =====================================================================
# Helper
# =====================================================================

def assert_tensor_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Assert that a tensor contains only finite values.
    """

    assert torch.isfinite(
        tensor
    ).all(), (
        f"{name} contains "
        "NaN or Inf values."
    )


# =====================================================================
# CNN stem
# =====================================================================

def test_cnn_stem_output_shape():

    model = CNNStem(
        in_channels=3,
        channels=(32, 64, 128),
    )

    x = torch.randn(
        1,
        3,
        128,
        128,
    )

    y = model(x)

    assert y.shape == (
        1,
        128,
        32,
        32,
    )

    assert_tensor_finite(
        y,
        "CNN stem output",
    )


# =====================================================================
# Complementarity module
# =====================================================================

def test_complementarity_module():

    module = ComplementarityEstimationModule(
        feature_channels=128,
        hidden_dim=256,
        descriptor_dim=128,
    )

    rgb = torch.randn(
        1,
        128,
        32,
        32,
    )

    thermal = torch.randn(
        1,
        128,
        32,
        32,
    )

    descriptor = module(
        rgb,
        thermal,
    )

    assert descriptor.shape == (
        1,
        128,
    )

    assert_tensor_finite(
        descriptor,
        "CEM descriptor",
    )


# =====================================================================
# Cross-modal interaction
# =====================================================================

def test_cross_modal_interaction():

    module = CrossModalInteraction(
        channels=128,
    )

    rgb = torch.randn(
        1,
        128,
        32,
        32,
    )

    thermal = torch.randn(
        1,
        128,
        32,
        32,
    )

    rgb_out, thermal_out = module(
        rgb,
        thermal,
    )

    assert rgb_out.shape == rgb.shape

    assert thermal_out.shape == thermal.shape

    assert_tensor_finite(
        rgb_out,
        "RGB cross-modal output",
    )

    assert_tensor_finite(
        thermal_out,
        "Thermal cross-modal output",
    )


# =====================================================================
# Vision Mamba encoder
# =====================================================================

def test_mamba_encoder_shape():

    encoder = VisionMambaEncoder(
        d_model=128,
        d_state=16,
        expand=2,
        num_blocks=1,
        num_scan_directions=4,
        spatial_size=(32, 32),
        backend="custom_ssm",
    )

    x = torch.randn(
        1,
        128,
        32,
        32,
    )

    y = encoder(x)

    assert y.shape == x.shape

    assert_tensor_finite(
        y,
        "Mamba encoder output",
    )


# =====================================================================
# Mamba backward
# =====================================================================

def test_mamba_encoder_backward():

    encoder = VisionMambaEncoder(
        d_model=128,
        d_state=16,
        expand=2,
        num_blocks=1,
        num_scan_directions=4,
        spatial_size=(16, 16),
        backend="custom_ssm",
    )

    x = torch.randn(
        1,
        128,
        16,
        16,
        requires_grad=True,
    )

    y = encoder(x)

    loss = y.mean()

    assert_tensor_finite(
        loss,
        "Mamba loss",
    )

    loss.backward()

    assert x.grad is not None

    assert_tensor_finite(
        x.grad,
        "Mamba input gradient",
    )

    for name, parameter in encoder.named_parameters():

        if parameter.grad is None:
            continue

        assert_tensor_finite(
            parameter.grad,
            f"Gradient: {name}",
        )


# =====================================================================
# Complete Stage-I model
# =====================================================================

@pytest.mark.parametrize(
    "image_size",
    [
        128,
        192,
        256,
    ],
)
def test_stage1_model_multiple_resolutions(
    image_size,
):

    config = make_test_config(
        image_size=image_size
    )

    model = Stage1Model(
        config=config
    )

    rgb = torch.randn(
        1,
        3,
        image_size,
        image_size,
    )

    thermal = torch.randn(
        1,
        1,
        image_size,
        image_size,
    )

    outputs = model(
        rgb,
        thermal,
    )

    # -------------------------------------------------------------
    # Required output keys
    # -------------------------------------------------------------

    required_keys = {
        "Fr",
        "Ft",
        "Fu",
        "complementarity",
        "rgb_cnn",
        "thermal_cnn",
    }

    assert required_keys.issubset(
        outputs.keys()
    )

    # -------------------------------------------------------------
    # Spatial size after the 1/4 CNN downsampling.
    # -------------------------------------------------------------

    expected_h = (
        image_size // 4
    )

    expected_w = (
        image_size // 4
    )

    assert outputs[
        "Fr"
    ].shape == (
        1,
        128,
        expected_h,
        expected_w,
    )

    assert outputs[
        "Ft"
    ].shape == (
        1,
        128,
        expected_h,
        expected_w,
    )

    assert outputs[
        "Fu"
    ].shape == (
        1,
        128,
        expected_h,
        expected_w,
    )

    # -------------------------------------------------------------
    # Complementarity descriptor
    # -------------------------------------------------------------

    assert outputs[
        "complementarity"
    ].shape == (
        1,
        128,
    )

    # -------------------------------------------------------------
    # Finite checks
    # -------------------------------------------------------------

    for key in required_keys:

        assert_tensor_finite(
            outputs[key],
            f"Model output: {key}",
        )


# =====================================================================
# Stage-I losses
# =====================================================================

def test_stage1_losses():

    config = make_test_config(
        image_size=128
    )

    criterion = Stage1Loss(
        config=config
    )

    batch_size = 1

    channels = 128

    height = 32

    width = 32

    rgb = torch.randn(
        batch_size,
        3,
        128,
        128,
    )

    thermal = torch.randn(
        batch_size,
        1,
        128,
        128,
    )

    Fr = torch.randn(
        batch_size,
        channels,
        height,
        width,
    )

    Ft = torch.randn(
        batch_size,
        channels,
        height,
        width,
    )

    Fu = torch.randn(
        batch_size,
        channels,
        height,
        width,
    )

    outputs = {
        "Fr": Fr,
        "Ft": Ft,
        "Fu": Fu,
    }

    losses = criterion(
        outputs,
        rgb=rgb,
        thermal=thermal,
    )

    required_keys = {
        "loss_total",
        "loss_structure",
        "loss_decorr",
        "loss_comp",
        "loss_info",
    }

    assert required_keys.issubset(
        losses.keys()
    )

    for key in required_keys:

        assert_tensor_finite(
            losses[key],
            f"Loss: {key}",
        )


# =====================================================================
# Full forward + backward
# =====================================================================

def test_stage1_full_backward():

    config = make_test_config(
        image_size=128
    )

    model = Stage1Model(
        config=config
    )

    criterion = Stage1Loss(
        config=config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4,
    )

    rgb = torch.randn(
        1,
        3,
        128,
        128,
    )

    thermal = torch.randn(
        1,
        1,
        128,
        128,
    )

    outputs = model(
        rgb,
        thermal,
    )

    losses = criterion(
        outputs,
        rgb=rgb,
        thermal=thermal,
    )

    total_loss = losses[
        "loss_total"
    ]

    assert_tensor_finite(
        total_loss,
        "Total Stage-I loss",
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss.backward()

    # -------------------------------------------------------------
    # Check gradients.
    # -------------------------------------------------------------

    gradient_count = 0

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        gradient_count += 1

        assert_tensor_finite(
            parameter.grad,
            f"Gradient: {name}",
        )

    assert (
        gradient_count > 0
    ), (
        "No gradients were produced "
        "during Stage-I backward."
    )

    # -------------------------------------------------------------
    # Optimizer update.
    # -------------------------------------------------------------

    optimizer.step()

    # -------------------------------------------------------------
    # Verify parameters remain finite.
    # -------------------------------------------------------------

    for name, parameter in model.named_parameters():

        assert_tensor_finite(
            parameter,
            f"Parameter after update: {name}",
        )


# =====================================================================
# Configurable image-size consistency
# =====================================================================

@pytest.mark.parametrize(
    "image_size",
    [
        128,
        192,
        256,
    ],
)
def test_config_image_size(
    image_size,
):

    config = make_test_config(
        image_size=image_size
    )

    assert (
        config.image_height
        == image_size
    )

    assert (
        config.image_width
        == image_size
    )

    expected_feature_size = (
        image_size // 4
    )

    assert (
        expected_feature_size
        > 0
    )


# =====================================================================
# No NaN / Inf in model parameters
# =====================================================================

def test_stage1_parameters_finite():

    config = make_test_config(
        image_size=128
    )

    model = Stage1Model(
        config=config
    )

    for name, parameter in model.named_parameters():

        assert_tensor_finite(
            parameter,
            f"Initial parameter: {name}",
        )


# =====================================================================
# CPU smoke test
# =====================================================================

def test_cpu_smoke():

    config = make_test_config(
        image_size=128
    )

    model = Stage1Model(
        config=config
    )

    model.eval()

    rgb = torch.randn(
        1,
        3,
        128,
        128,
    )

    thermal = torch.randn(
        1,
        1,
        128,
        128,
    )

    with torch.no_grad():

        outputs = model(
            rgb,
            thermal,
        )

    assert outputs is not None

    assert "Fu" in outputs

    assert_tensor_finite(
        outputs["Fu"],
        "CPU smoke-test fused feature",
    )