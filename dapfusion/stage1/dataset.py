"""
Dataset Loader for Paired RGB-Thermal Images (LLVIP and General Datasets)
Reference: DAPFusion Framework - Stage I: Multi-Modal Feature Representation Backbone
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PairedRGBThermalDataset(Dataset):
    """Generic Paired RGB-Thermal Dataset."""

    def __init__(
        self,
        rgb_dir: Union[str, Path],
        thermal_dir: Union[str, Path],
        image_size: Tuple[int, int] = (256, 256),
        transform_rgb: Optional[Callable] = None,
        transform_thermal: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.rgb_dir = Path(rgb_dir)
        self.thermal_dir = Path(thermal_dir)
        self.image_size = image_size

        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        if not self.thermal_dir.exists():
            raise FileNotFoundError(f"Thermal directory not found: {self.thermal_dir}")

        rgb_files = {
            f for f in os.listdir(self.rgb_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        }
        thermal_files = {
            f for f in os.listdir(self.thermal_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        }

        self.filenames: List[str] = sorted(list(rgb_files.intersection(thermal_files)))
        if not self.filenames:
            raise ValueError(
                f"No matching image pairs found between {self.rgb_dir} and {self.thermal_dir}."
            )

        self.transform_rgb = transform_rgb or transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.transform_thermal = transform_thermal or transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        rgb_path = self.rgb_dir / filename
        thermal_path = self.thermal_dir / filename

        with Image.open(rgb_path) as img:
            rgb_img = img.convert("RGB")
        rgb_tensor = self.transform_rgb(rgb_img)

        with Image.open(thermal_path) as img:
            thermal_img = img.convert("L")
        thermal_tensor = self.transform_thermal(thermal_img)

        return rgb_tensor, thermal_tensor


def create_llvip_dataloader(
    root_dir: Union[str, Path],
    split: str = "train",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (256, 256),
) -> DataLoader:
    """Factory function to construct a DataLoader for the LLVIP dataset."""
    root_path = Path(root_dir)
    rgb_dir = root_path / "visible" / split
    thermal_dir = root_path / "infrared" / split

    dataset = PairedRGBThermalDataset(
        rgb_dir=rgb_dir,
        thermal_dir=thermal_dir,
        image_size=image_size,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
