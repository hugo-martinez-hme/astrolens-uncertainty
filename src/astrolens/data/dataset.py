"""PyTorch Dataset for astronomical FITS cutouts with strict endianness and NaN validation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset


def sanitize_byte_order(array: np.ndarray) -> np.ndarray:
    """Convert big-endian FITS arrays ('>f4', '>f8') into native C-contiguous float32.

    FITS stores floating-point data in IEEE 754 big-endian order, whereas modern x86_64/ARM64
    CPUs and PyTorch tensors require native little-endian buffers.

    Args:
        array: Raw NumPy array loaded from an AstroPy FITS HDU.

    Returns:
        Native-endian, writable, C-contiguous np.float32 array.
    """
    native_f32 = np.dtype("=f4")
    if array.dtype != native_f32 or not array.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(array, dtype=native_f32)
    if not array.flags["WRITEABLE"]:
        return array.copy()
    return array


class FitsLensDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch Dataset for synthetic or real (Euclid/HST) strong lensing FITS cutouts.

    Loads 2D (H, W) or 3D (C, H, W) science arrays from FITS files, normalizes byte order
    to native float32, validates finite flux values, and applies modular PyTorch transforms.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        file_paths: list[Path] | None = None,
        hdu_index: int = 0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        label_key: str = "CLASS_ID",
    ) -> None:
        """Initialize the FITS dataset.

        Args:
            root_dir: Directory containing .fits files (discovered recursively, sorted).
            file_paths: Explicit list of FITS paths (overrides root_dir if provided).
            hdu_index: FITS HDU index containing the image array.
            transform: Optional callable mapping a (C, H, W) float32 tensor to a tensor.
            label_key: Header keyword storing integer class label (-1 if absent).

        Raises:
            ValueError: If neither root_dir nor file_paths is given, or no FITS files exist.
        """
        if file_paths is not None:
            self.file_paths = [Path(p) for p in file_paths]
        elif root_dir is not None:
            self.file_paths = sorted(Path(root_dir).rglob("*.fits"))
        else:
            raise ValueError("Either 'root_dir' or 'file_paths' must be provided.")

        if not self.file_paths:
            raise ValueError("No .fits files found for FitsLensDataset.")

        self.hdu_index = hdu_index
        self.transform = transform
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.file_paths[index]

        with fits.open(path, memmap=False) as hdul:
            hdu = hdul[self.hdu_index]
            raw_data = hdu.data
            header = hdu.header

            if raw_data is None:
                raise ValueError(f"HDU {self.hdu_index} in '{path.name}' contains no image data.")

            clean_array = sanitize_byte_order(raw_data)
            if not np.all(np.isfinite(clean_array)):
                raise ValueError(f"Non-finite values (NaN/Inf) detected in FITS file '{path}'.")

            if clean_array.ndim == 2:
                clean_array = clean_array[np.newaxis, :, :]  # (1, H, W)
            elif clean_array.ndim != 3:
                raise ValueError(
                    f"Expected 2D (H, W) or 3D (C, H, W) FITS array, got {clean_array.shape}."
                )

            label = int(header.get(self.label_key, -1))

        tensor = torch.from_numpy(clean_array)
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label
