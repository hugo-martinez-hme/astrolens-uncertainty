"""Axiomatic, physical, and integration tests for Module 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from astropy.io import fits
from astropy.wcs import WCS
from torch.utils.data import DataLoader

from astrolens.data import (
    ArcsinhStretch,
    DihedralD4Transform,
    FitsLensDataset,
    generate_synthetic_dataset,
    sanitize_byte_order,
)


def test_synthetic_fits_generation_and_wcs_validity(tmp_path: Path) -> None:
    """Verify synthetic FITS files store valid WCS headers, metadata, and big-endian floats."""
    files = generate_synthetic_dataset(output_dir=tmp_path, samples_per_class=2, seed=123)
    assert len(files) == 6

    with fits.open(files[0], memmap=False) as hdul:
        hdu = hdul[0]
        assert hdu.data.shape == (64, 64)
        # FITS standard serializes IEEE float32 in big-endian ('>f4')
        assert hdu.data.dtype.byteorder == ">"
        assert hdu.header["CLASS"] in ("smooth", "cdm_subhalo", "anomaly")
        assert "PSFSIGMA" in hdu.header
        assert "EXPTIME" in hdu.header
        assert "RDNOISE" in hdu.header

        wcs = WCS(hdu.header)
        assert wcs.has_celestial


def test_endianness_sanitization_and_nan_rejection(tmp_path: Path) -> None:
    """Ensure big-endian (>f4) FITS arrays convert cleanly and NaN/Inf files raise ValueError."""
    raw_be = np.arange(16, dtype=">f4").reshape(4, 4)
    assert raw_be.dtype.byteorder == ">"
    clean = sanitize_byte_order(raw_be)
    assert clean.dtype == np.float32
    assert clean.dtype.byteorder in ("=", "|")
    assert clean.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(clean, raw_be.astype(np.float32))

    # Corrupt FITS file containing a NaN pixel must be rejected immediately
    corrupt_arr = np.ones((16, 16), dtype=np.float32)
    corrupt_arr[3, 5] = np.nan
    corrupt_path = tmp_path / "corrupt_nan.fits"
    fits.PrimaryHDU(data=corrupt_arr).writeto(corrupt_path)

    dataset = FitsLensDataset(file_paths=[corrupt_path])
    with pytest.raises(ValueError, match="Non-finite"):
        _ = dataset[0]


def test_arcsinh_stability_monotonicity_and_regimes() -> None:
    """Verify arcsinh has zero NaNs on negative sky, is strictly monotonic, and invertible."""
    beta = 10.0
    stretch = ArcsinhStretch(beta=beta)

    # Full physical dynamic range: negative sky noise up to saturated galaxy core
    x = torch.linspace(-50.0, 50_000.0, 1000, dtype=torch.float32)
    y = stretch(x)

    assert torch.all(torch.isfinite(y))
    # Strict monotonicity: y[i+1] > y[i] everywhere
    assert torch.all(torch.diff(y) > 0.0)

    # Low-SNR linear Taylor regime: arcsinh(x / beta) ~ x / beta
    faint = torch.tensor([-1e-2, 0.0, 1e-2], dtype=torch.float32)
    torch.testing.assert_close(stretch(faint), faint / beta, rtol=1e-5, atol=1e-7)

    # High-SNR logarithmic regime: arcsinh(x / beta) ~ ln(2x / beta)
    bright = torch.tensor([1e5], dtype=torch.float32)
    torch.testing.assert_close(
        stretch(bright), torch.log(2.0 * bright / beta), rtol=1e-4, atol=1e-4
    )

    # Exact bijective inverse
    reconstructed = stretch.inverse(y)
    torch.testing.assert_close(reconstructed, x, rtol=1e-4, atol=1e-3)


def test_dihedral_d4_closure_exact_inverse_and_l2_invariance() -> None:
    """Verify all 8 D4 symmetries are distinct, preserve L2 norm, and invert with zero error."""
    torch.manual_seed(7)
    x = torch.randn(1, 32, 32, dtype=torch.float32)
    orig_norm = torch.linalg.vector_norm(x)

    orbit = DihedralD4Transform.orbit_expand(x)
    assert orbit.shape == (8, 1, 32, 32)

    # 1. All 8 group actions produce distinct states on a generic asymmetric field
    for i in range(8):
        for j in range(i + 1, 8):
            assert not torch.allclose(orbit[i], orbit[j])

    # 2. L2 norm conservation and bit-exact inversion g_k^{-1}(g_k(x)) == x
    for k in range(8):
        det_transform = DihedralD4Transform(k=k)
        transformed = det_transform(x)
        assert transformed.shape == x.shape
        torch.testing.assert_close(torch.linalg.vector_norm(transformed), orig_norm)

        restored = DihedralD4Transform.invert_element(transformed, k)
        assert torch.equal(restored, x), f"Bit-exact inversion failed for k={k}."


def test_fits_dataset_and_dataloader_pipeline(tmp_path: Path) -> None:
    """Verify end-to-end FitsLensDataset + transforms + PyTorch DataLoader batching."""
    generate_synthetic_dataset(output_dir=tmp_path, samples_per_class=4, seed=99)

    pipeline = nn.Sequential(ArcsinhStretch(beta=10.0), DihedralD4Transform(p=1.0))
    dataset = FitsLensDataset(root_dir=tmp_path, transform=pipeline)
    assert len(dataset) == 12

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch_images, batch_labels = next(iter(loader))

    assert batch_images.shape == (4, 1, 64, 64)
    assert batch_images.dtype == torch.float32
    assert torch.all(torch.isfinite(batch_images))
    assert batch_labels.shape == (4,)
    assert set(batch_labels.tolist()).issubset({0, 1, 2})
