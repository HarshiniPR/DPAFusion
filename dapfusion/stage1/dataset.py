"""
Dataset Loader for Paired RGB-Thermal Images
============================================

Reference:
    DAPFusion Framework - Stage I:
    Multi-Modal Feature Representation Backbone

Primary dataset:
    LLVIP

Expected LLVIP structure:

    LLVIP/
    ├── visible/
    │   ├── train/
    │   └── test/
    │
    └── infrared/
        ├── train/
        └── test/

The loader:

    1. Finds matching RGB/Thermal filenames.
    2. Loads RGB images as 3-channel RGB.
    3. Loads Thermal images as single-channel grayscale.
    4. Resizes both modalities to a configurable resolution.
    5. Converts them to PyTorch tensors.

Supported image-size experiments:

    128 x 128
    192 x 192
    256 x 256

No image size is hard-coded in the model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# Project-root import handling
# ---------------------------------------------------------------------
_project_root = str(Path(__file__).resolve().parents[2])

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =====================================================================
# Type aliases
# =====================================================================

PathLike = Union[str, Path]

ImageSize = Union[
    int,
    Tuple[int, int],
]


# =====================================================================
# Helper functions
# =====================================================================

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


def normalize_image_size(
    image_size: ImageSize,
) -> Tuple[int, int]:
    """
    Normalize image-size input to:

        (height, width)

    Examples:

        256
            -> (256, 256)

        (128, 128)
            -> (128, 128)

        [192, 192]
            -> (192, 192)
    """

    if isinstance(image_size, int):

        if image_size <= 0:
            raise ValueError(
                f"image_size must be positive, got {image_size}"
            )

        return image_size, image_size

    if not isinstance(image_size, (tuple, list)):
        raise TypeError(
            "image_size must be an int or a "
            "(height, width) tuple."
        )

    if len(image_size) != 2:
        raise ValueError(
            "image_size must contain exactly two values: "
            "(height, width)."
        )

    height = int(image_size[0])
    width = int(image_size[1])

    if height <= 0 or width <= 0:
        raise ValueError(
            f"Image dimensions must be positive, "
            f"got {height}x{width}."
        )

    return height, width


def list_image_files(
    directory: Path,
) -> List[str]:
    """
    Return sorted image filenames from a directory.

    Only files with supported image extensions are included.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, got: {directory}"
        )

    filenames = [
        filename
        for filename in os.listdir(directory)
        if filename.lower().endswith(
            SUPPORTED_EXTENSIONS
        )
    ]

    return sorted(filenames)


# =====================================================================
# Paired RGB-Thermal Dataset
# =====================================================================

class PairedRGBThermalDataset(Dataset):
    """
    Generic paired RGB-Thermal dataset.

    Each sample consists of:

        RGB:
            (3, H, W)

        Thermal:
            (1, H, W)

    where H and W are determined by ``image_size``.

    Args:
        rgb_dir:
            Directory containing RGB/visible images.

        thermal_dir:
            Directory containing thermal/infrared images.

        image_size:
            Target image size.

            Can be:

                128
                192
                256

            or an explicit tuple such as:

                (256, 256)

        transform_rgb:
            Optional custom RGB transformation.

        transform_thermal:
            Optional custom Thermal transformation.

        strict_pairing:
            If True, raise an error when one modality contains files
            without a corresponding file in the other modality.

            Default=False because LLVIP and other datasets may contain
            modality-specific files that are not necessarily part of
            the current paired experiment.

        return_filename:
            If True, return the filename together with the tensors.

            Default=False to preserve the original training API.
    """

    def __init__(
        self,
        rgb_dir: PathLike,
        thermal_dir: PathLike,
        image_size: ImageSize = (256, 256),
        transform_rgb: Optional[Callable] = None,
        transform_thermal: Optional[Callable] = None,
        strict_pairing: bool = False,
        return_filename: bool = False,
    ) -> None:

        super().__init__()

        # -------------------------------------------------------------
        # Paths
        # -------------------------------------------------------------
        self.rgb_dir = Path(rgb_dir)
        self.thermal_dir = Path(thermal_dir)

        # -------------------------------------------------------------
        # Image size
        # -------------------------------------------------------------
        self.image_size = normalize_image_size(
            image_size
        )

        # -------------------------------------------------------------
        # Options
        # -------------------------------------------------------------
        self.strict_pairing = strict_pairing
        self.return_filename = return_filename

        # -------------------------------------------------------------
        # Validate directories
        # -------------------------------------------------------------
        if not self.rgb_dir.exists():
            raise FileNotFoundError(
                f"RGB directory not found: {self.rgb_dir}"
            )

        if not self.thermal_dir.exists():
            raise FileNotFoundError(
                f"Thermal directory not found: "
                f"{self.thermal_dir}"
            )

        # -------------------------------------------------------------
        # Find image files
        # -------------------------------------------------------------
        rgb_files = set(
            list_image_files(
                self.rgb_dir
            )
        )

        thermal_files = set(
            list_image_files(
                self.thermal_dir
            )
        )

        # -------------------------------------------------------------
        # Determine unmatched files
        # -------------------------------------------------------------
        rgb_only = sorted(
            rgb_files - thermal_files
        )

        thermal_only = sorted(
            thermal_files - rgb_files
        )

        # -------------------------------------------------------------
        # Optional strict pairing check
        # -------------------------------------------------------------
        if strict_pairing and (
            rgb_only or thermal_only
        ):
            raise ValueError(
                "RGB/Thermal directories are not perfectly paired.\n"
                f"RGB-only files: {len(rgb_only)}\n"
                f"Thermal-only files: {len(thermal_only)}\n"
                f"RGB directory: {self.rgb_dir}\n"
                f"Thermal directory: {self.thermal_dir}"
            )

        # -------------------------------------------------------------
        # Only use filenames present in BOTH modalities.
        # -------------------------------------------------------------
        self.filenames: List[str] = sorted(
            rgb_files.intersection(
                thermal_files
            )
        )

        if not self.filenames:
            raise ValueError(
                "No matching RGB/Thermal image pairs found.\n"
                f"RGB directory: {self.rgb_dir}\n"
                f"Thermal directory: {self.thermal_dir}"
            )

        # -------------------------------------------------------------
        # Default RGB transformation
        # -------------------------------------------------------------
        #
        # RGB:
        #
        #   PIL RGB
        #       ->
        #   Resize
        #       ->
        #   Tensor
        #
        # Result:
        #
        #   (3, H, W)
        #
        # No ImageNet normalization is applied because the Stage-I
        # feature-learning pipeline does not require it and applying
        # different pretrained-model normalization would alter the
        # input distribution.
        # -------------------------------------------------------------
        self.transform_rgb = (
            transform_rgb
            if transform_rgb is not None
            else transforms.Compose(
                [
                    transforms.Resize(
                        self.image_size,
                        interpolation=(
                            transforms.InterpolationMode.BILINEAR
                        ),
                    ),
                    transforms.ToTensor(),
                ]
            )
        )

        # -------------------------------------------------------------
        # Default Thermal transformation
        # -------------------------------------------------------------
        #
        # Thermal:
        #
        #   PIL grayscale ("L")
        #       ->
        #   Resize
        #       ->
        #   Tensor
        #
        # Result:
        #
        #   (1, H, W)
        # -------------------------------------------------------------
        self.transform_thermal = (
            transform_thermal
            if transform_thermal is not None
            else transforms.Compose(
                [
                    transforms.Resize(
                        self.image_size,
                        interpolation=(
                            transforms.InterpolationMode.BILINEAR
                        ),
                    ),
                    transforms.ToTensor(),
                ]
            )
        )

    # -----------------------------------------------------------------
    # Dataset length
    # -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.filenames)

    # -----------------------------------------------------------------
    # Dataset sample
    # -----------------------------------------------------------------

    def __getitem__(
        self,
        idx: int,
    ):
        """
        Load one paired RGB-Thermal sample.

        Returns:

            if return_filename=False:

                rgb_tensor, thermal_tensor

            if return_filename=True:

                rgb_tensor, thermal_tensor, filename
        """

        if idx < 0:
            idx += len(self.filenames)

        if idx < 0 or idx >= len(self.filenames):
            raise IndexError(
                f"Dataset index out of range: {idx}"
            )

        filename = self.filenames[idx]

        rgb_path = self.rgb_dir / filename
        thermal_path = self.thermal_dir / filename

        # -------------------------------------------------------------
        # Load RGB
        # -------------------------------------------------------------
        try:

            with Image.open(rgb_path) as img:

                # Explicitly convert to RGB so that the output is
                # guaranteed to have exactly three channels.
                rgb_img = img.convert("RGB")

                rgb_tensor = self.transform_rgb(
                    rgb_img
                )

        except (
            OSError,
            UnidentifiedImageError,
        ) as exc:

            raise RuntimeError(
                f"Failed to load RGB image: {rgb_path}"
            ) from exc

        # -------------------------------------------------------------
        # Load Thermal
        # -------------------------------------------------------------
        try:

            with Image.open(thermal_path) as img:

                # LLVIP infrared images may be stored as RGB-looking
                # image files, but Stage I uses a single thermal
                # intensity channel.
                thermal_img = img.convert("L")

                thermal_tensor = self.transform_thermal(
                    thermal_img
                )

        except (
            OSError,
            UnidentifiedImageError,
        ) as exc:

            raise RuntimeError(
                f"Failed to load Thermal image: "
                f"{thermal_path}"
            ) from exc

        # -------------------------------------------------------------
        # Validate tensor dimensions.
        # -------------------------------------------------------------
        if rgb_tensor.ndim != 3:
            raise RuntimeError(
                f"RGB transform must produce a 3D tensor "
                f"(C,H,W), got {tuple(rgb_tensor.shape)}"
            )

        if thermal_tensor.ndim != 3:
            raise RuntimeError(
                f"Thermal transform must produce a 3D tensor "
                f"(C,H,W), got {tuple(thermal_tensor.shape)}"
            )

        if rgb_tensor.shape[0] != 3:
            raise RuntimeError(
                f"RGB image must have 3 channels, "
                f"got {rgb_tensor.shape[0]} "
                f"for {rgb_path}"
            )

        if thermal_tensor.shape[0] != 1:
            raise RuntimeError(
                f"Thermal image must have 1 channel, "
                f"got {thermal_tensor.shape[0]} "
                f"for {thermal_path}"
            )

        expected_h, expected_w = self.image_size

        if rgb_tensor.shape[-2:] != (
            expected_h,
            expected_w,
        ):
            raise RuntimeError(
                "RGB transform produced an unexpected "
                f"spatial size: {tuple(rgb_tensor.shape)}"
            )

        if thermal_tensor.shape[-2:] != (
            expected_h,
            expected_w,
        ):
            raise RuntimeError(
                "Thermal transform produced an unexpected "
                f"spatial size: {tuple(thermal_tensor.shape)}"
            )

        # -------------------------------------------------------------
        # Return
        # -------------------------------------------------------------
        if self.return_filename:
            return (
                rgb_tensor,
                thermal_tensor,
                filename,
            )

        return (
            rgb_tensor,
            thermal_tensor,
        )


# =====================================================================
# LLVIP DataLoader Factory
# =====================================================================

def create_llvip_dataloader(
    root_dir: PathLike,
    split: str = "train",
    batch_size: int = 1,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    image_size: ImageSize = (256, 256),
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for LLVIP.

    Expected root directory:

        LLVIP/

    containing:

        visible/train
        visible/test
        infrared/train
        infrared/test

    Args:
        root_dir:
            Root LLVIP directory.

        split:
            Dataset split.

            Common values:

                "train"
                "test"

        batch_size:
            Physical GPU batch size.

            Default = 1 for memory-safe Stage-I development.

        shuffle:
            Whether to shuffle the dataset.

            If None:
                True for training,
                False otherwise.

        num_workers:
            Number of DataLoader worker processes.

        image_size:
            Target spatial resolution.

        pin_memory:
            Whether to pin CPU memory.

            If None, automatically enables it when CUDA is
            available.

        drop_last:
            Whether to drop an incomplete final batch.

        persistent_workers:
            Whether DataLoader workers remain alive between epochs.

    Returns:
        torch.utils.data.DataLoader
    """

    # -------------------------------------------------------------
    # Validate batch size
    # -------------------------------------------------------------
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}"
        )

    # -------------------------------------------------------------
    # Normalize split
    # -------------------------------------------------------------
    split = str(split).strip().lower()

    if not split:
        raise ValueError(
            "split cannot be empty."
        )

    # -------------------------------------------------------------
    # Normalize image size
    # -------------------------------------------------------------
    image_size = normalize_image_size(
        image_size
    )

    # -------------------------------------------------------------
    # Root
    # -------------------------------------------------------------
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(
            f"LLVIP root directory not found: "
            f"{root_path}"
        )

    # -------------------------------------------------------------
    # LLVIP modality directories
    # -------------------------------------------------------------
    rgb_dir = (
        root_path
        / "visible"
        / split
    )

    thermal_dir = (
        root_path
        / "infrared"
        / split
    )

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------
    dataset = PairedRGBThermalDataset(
        rgb_dir=rgb_dir,
        thermal_dir=thermal_dir,
        image_size=image_size,
    )

    # -------------------------------------------------------------
    # Shuffle default
    # -------------------------------------------------------------
    if shuffle is None:
        shuffle = split == "train"

    # -------------------------------------------------------------
    # pin_memory default
    # -------------------------------------------------------------
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # -------------------------------------------------------------
    # persistent_workers only works when workers > 0.
    # -------------------------------------------------------------
    if num_workers == 0:
        persistent_workers = False

    # -------------------------------------------------------------
    # DataLoader
    # -------------------------------------------------------------
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )

    return loader


# =====================================================================
# Standalone verification
# =====================================================================

if __name__ == "__main__":

    print(
        "Dataset module imported successfully."
    )

    print(
        f"Supported image extensions: "
        f"{SUPPORTED_EXTENSIONS}"
    )

    for size in (
        128,
        192,
        256,
    ):

        normalized = normalize_image_size(size)

        assert normalized == (
            size,
            size,
        )

        print(
            f"Image size {size}: "
            f"{normalized}"
        )

    print(
        "Image-size utilities verified successfully."
    )