"""Simplified morphological strong gravitational lens generator in standard FITS format.

Simulates 64x64 Euclid-VIS-like cutouts (0.1 arcsec/pixel, EXPTIME=565s, GAIN=3.1 e-/ADU,
RDNOISE=4.5 e-) across three predefined morphological classes:
    0 ('smooth'): Smooth elliptical lens + unperturbed Einstein ring/arc.
    1 ('cdm_subhalo'): Lens + Einstein ring perturbed by localized CDM subhalo kinks/knots.
    2 ('anomaly'): Compound lens morphology (double rings and asymmetric companion).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from astropy.wcs import WCS

CLASS_NAMES: tuple[str, str, str] = ("smooth", "cdm_subhalo", "anomaly")


@dataclass(frozen=True)
class InstrumentConfig:
    """Simplified detector and optical configuration inspired by Euclid VIS."""

    image_size: int = 64
    pixel_scale_arcsec: float = 0.1
    exptime_s: float = 565.0
    gain_e_per_adu: float = 3.1
    read_noise_e: float = 4.5
    sky_rate_e_per_s: float = 0.25
    psf_sigma_px: float = 1.1


def convolve_gaussian_psf(image: np.ndarray, psf_sigma: float) -> np.ndarray:
    r"""Convolve a 2D flux map with a normalized isotropic Gaussian PSF.

    Implements :math:`(S * K_{\text{PSF}})_{ij}` in float64 with reflection padding to
    preserve constant fields (:math:`\sum K_{\text{PSF}} = 1`) without external SciPy
    dependencies or FFT wrap-around artifacts. Reflection boundaries do not guarantee
    exact conservation of the total flux over a finite cutout.

    Args:
        image: 2D array of shape (H, W) in physical rate units (e-/s).
        psf_sigma: Standard deviation of the Gaussian PSF in pixels.

    Returns:
        PSF-convolved 2D float64 array of shape (H, W).
    """
    if psf_sigma <= 0.0:
        return image.astype(np.float64, copy=True)

    radius = max(1, int(np.ceil(4.0 * psf_sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / (psf_sigma**2))
    kernel /= np.sum(kernel)

    img_t = torch.from_numpy(image.astype(np.float64))[None, None, :, :]
    ker_t = torch.from_numpy(kernel)[None, None, :, :]
    padded = F.pad(img_t, (radius, radius, radius, radius), mode="reflect")
    convolved = F.conv2d(padded, ker_t)
    return convolved.squeeze(0).squeeze(0).numpy()


def _elliptical_grid(
    size: int,
    dx: float,
    dy: float,
    axis_ratio: float,
    angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute elliptical radius and azimuthal angle grids centered at (dx, dy)."""
    half = (size - 1) / 2.0
    y, x = np.meshgrid(
        np.arange(size, dtype=np.float64) - half - dy,
        np.arange(size, dtype=np.float64) - half - dx,
        indexing="ij",
    )
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    xr = x * cos_a + y * sin_a
    yr = -x * sin_a + y * cos_a
    r_ell = np.sqrt((xr**2) * axis_ratio + (yr**2) / axis_ratio)
    theta = np.arctan2(yr, xr)
    return r_ell, theta


def simulate_lens_image(
    class_id: int,
    rng: np.random.Generator,
    config: InstrumentConfig | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    r"""Simulate a single noisy strong gravitational lensing frame in ADU.

    Instrumental forward model (applied to a parametrized morphology):
    .. math::
        \mu_{ij} = \tau \left( [S * K_{\text{PSF}}]_{ij} + \lambda_{\text{sky}} \right),
        \quad N_{ij} \sim \text{Poisson}(\mu_{ij}),
        \quad I_{ij} = \frac{N_{ij} + \epsilon_{ij} - \tau \lambda_{\text{sky}}}{g}

    Args:
        class_id: Morphological class index (0=smooth, 1=cdm_subhalo, 2=anomaly).
        rng: NumPy random generator controlling the sampling sequence.
        config: Detector and optical parameters.

    Returns:
        Tuple of (image_adu_float32, metadata_dict).
    """
    if class_id not in (0, 1, 2):
        raise ValueError(f"class_id must be in {{0, 1, 2}}, got {class_id}.")

    cfg = config or InstrumentConfig()
    size = cfg.image_size

    # Small sub-pixel centering jitter and foreground lens geometry
    dx = float(rng.uniform(-1.5, 1.5))
    dy = float(rng.uniform(-1.5, 1.5))
    q_lens = float(rng.uniform(0.76, 0.95))
    phi_lens = float(rng.uniform(0.0, np.pi))

    r_lens, _ = _elliptical_grid(size, dx, dy, q_lens, phi_lens)

    # 1. Foreground elliptical lens galaxy (compact Sersic-like core, ~100x brighter than arc)
    peak_core = float(rng.uniform(120.0, 190.0))  # e-/s/px
    scale_core = float(rng.uniform(1.8, 2.3))
    flux_rate = peak_core * np.exp(-((r_lens / scale_core) ** 1.05))

    # 2. Primary Einstein ring / gravitational arc (background lensed source)
    q_ring = float(rng.uniform(0.84, 0.98))
    phi_ring = float(rng.uniform(0.0, np.pi))
    r_ring, theta = _elliptical_grid(size, dx, dy, q_ring, phi_ring)

    r_einstein = float(rng.uniform(13.5, 18.5))
    arc_width = float(rng.uniform(1.2, 1.7))
    arc_amp = float(rng.uniform(1.4, 2.4))  # e-/s/px (still ~2% of core peak in linear ADU)

    phase_quad = float(rng.uniform(0.0, np.pi))
    phase_dip = float(rng.uniform(0.0, 2.0 * np.pi))
    azimuthal_mod = np.clip(
        1.0 + 0.45 * np.cos(2.0 * (theta - phase_quad)) + 0.25 * np.cos(theta - phase_dip),
        0.15,
        2.0,
    )

    if class_id == 0:
        # Smooth dark matter halo (no substructure)
        ring_flux = (
            arc_amp * azimuthal_mod * np.exp(-0.5 * ((r_ring - r_einstein) / arc_width) ** 2)
        )
        flux_rate += ring_flux

    elif class_id == 1:
        # Cold Dark Matter (CDM) subhalo: localized radial deflection kink + flux anomaly knots
        n_subhalos = int(rng.integers(2, 4))
        delta_r = np.zeros_like(r_ring)
        knot_flux = np.zeros_like(r_ring)

        for _ in range(n_subhalos):
            sub_theta = float(rng.uniform(-np.pi, np.pi))
            d_theta = np.arctan2(np.sin(theta - sub_theta), np.cos(theta - sub_theta))
            kink_amp = float(rng.uniform(1.6, 2.8)) * float(rng.choice([-1.0, 1.0]))
            kink_width = float(rng.uniform(0.18, 0.30))
            delta_r += kink_amp * np.exp(-0.5 * (d_theta / kink_width) ** 2)

            # Compact lensed subhalo brightness anomaly right along the perturbed arc
            sub_r = r_einstein + 0.5 * kink_amp
            sx = sub_r * np.cos(sub_theta) / np.sqrt(q_ring)
            sy = sub_r * np.sin(sub_theta) * np.sqrt(q_ring)
            half = (size - 1) / 2.0
            y_grid, x_grid = np.meshgrid(
                np.arange(size, dtype=np.float64) - half - dy,
                np.arange(size, dtype=np.float64) - half - dx,
                indexing="ij",
            )
            cos_p, sin_p = np.cos(phi_ring), np.sin(phi_ring)
            gx = sx * cos_p - sy * sin_p
            gy = sx * sin_p + sy * cos_p
            knot_sigma = float(rng.uniform(1.1, 1.6))
            knot_amp = float(rng.uniform(1.1, 2.2)) * arc_amp
            knot_flux += knot_amp * np.exp(
                -0.5 * ((x_grid - gx) ** 2 + (y_grid - gy) ** 2) / (knot_sigma**2)
            )

        perturbed_radius = r_einstein + delta_r
        ring_flux = (
            arc_amp * azimuthal_mod * np.exp(-0.5 * ((r_ring - perturbed_radius) / arc_width) ** 2)
        )
        flux_rate += ring_flux + knot_flux

    else:
        # Predefined anomaly morphology: double rings and asymmetric companion
        r_inner = float(rng.uniform(10.0, 12.5))
        r_outer = float(rng.uniform(20.0, 23.5))
        r_outer_grid, theta_out = _elliptical_grid(
            size, dx, dy, float(rng.uniform(0.75, 0.90)), phi_ring + np.pi / 3.0
        )

        inner_ring = (
            arc_amp * azimuthal_mod * np.exp(-0.5 * ((r_ring - r_inner) / (arc_width * 0.9)) ** 2)
        )
        outer_mod = np.clip(0.8 + 0.7 * np.cos(theta_out - phase_dip), 0.1, 1.8)
        outer_ring = (
            (0.85 * arc_amp)
            * outer_mod
            * np.exp(-0.5 * ((r_outer_grid - r_outer) / (arc_width * 1.1)) ** 2)
        )

        # Asymmetric luminous companion / tidal tail
        comp_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        comp_dist = float(rng.uniform(15.0, 21.0))
        cx = comp_dist * np.cos(comp_angle)
        cy = comp_dist * np.sin(comp_angle)
        r_comp, _ = _elliptical_grid(size, dx + cx, dy + cy, 0.65, comp_angle)
        companion = float(rng.uniform(6.0, 12.0)) * np.exp(-0.5 * (r_comp / 2.2) ** 2)
        flux_rate += inner_ring + outer_ring + companion

    # 3. Optical PSF convolution + sky background
    psf_sigma = float(rng.uniform(0.95, 1.25)) * cfg.psf_sigma_px
    convolved_rate = convolve_gaussian_psf(flux_rate, psf_sigma) + cfg.sky_rate_e_per_s
    expected_electrons = np.clip(convolved_rate * cfg.exptime_s, 0.0, None)

    # 4. Quantum Poisson shot noise + Gaussian CCD read noise
    shot_electrons = rng.poisson(expected_electrons).astype(np.float64)
    read_noise = rng.normal(0.0, cfg.read_noise_e, size=(size, size))

    # 5. Sky subtraction and electron-to-ADU conversion (preserves legitimate negative sky pixels)
    sky_electrons = cfg.sky_rate_e_per_s * cfg.exptime_s
    image_adu = (shot_electrons + read_noise - sky_electrons) / cfg.gain_e_per_adu

    meta: dict[str, float | int | str] = {
        "CLASS": CLASS_NAMES[class_id],
        "CLASS_ID": int(class_id),
        "PSFSIGMA": round(psf_sigma, 4),
        "EXPTIME": cfg.exptime_s,
        "RDNOISE": cfg.read_noise_e,
        "GAIN": cfg.gain_e_per_adu,
        "SKYLVL": round(sky_electrons / cfg.gain_e_per_adu, 4),
    }
    return image_adu.astype(np.float32), meta


def build_wcs_header(
    size: int,
    pixel_scale_arcsec: float,
    ra_deg: float,
    dec_deg: float,
    meta: dict[str, float | int | str],
) -> fits.Header:
    """Create a FITS-4.0-compliant header with valid TAN-projection WCS and physical metadata."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [size / 2.0 + 0.5, size / 2.0 + 0.5]
    wcs.wcs.cdelt = np.array([-pixel_scale_arcsec / 3600.0, pixel_scale_arcsec / 3600.0])
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]

    header = wcs.to_header()
    header["TELESCOP"] = ("EUCLID-SIM", "Simulated telescope mission")
    header["INSTRUME"] = ("VIS-MOCK", "Simulated optical imager")
    for key, value in meta.items():
        header[key] = value
    return header


def generate_synthetic_dataset(
    output_dir: str | Path,
    samples_per_class: int = 30,
    seed: int = 42,
    config: InstrumentConfig | None = None,
) -> list[Path]:
    """Generate a reproducible balanced dataset of synthetic strong lensing FITS files.

    Args:
        output_dir: Directory where .fits files will be written.
        samples_per_class: Number of frames to generate per morphological class.
        seed: Random seed; verify generated file hashes in the target software environment.
        config: Optional detector/optics configuration.

    Returns:
        List of generated FITS file paths.
    """
    cfg = config or InstrumentConfig()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    created_files: list[Path] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        for idx in range(samples_per_class):
            image, meta = simulate_lens_image(class_id=class_id, rng=rng, config=cfg)
            ra_deg = float(rng.uniform(0.0, 360.0))
            dec_deg = float(rng.uniform(-70.0, 70.0))
            header = build_wcs_header(
                size=cfg.image_size,
                pixel_scale_arcsec=cfg.pixel_scale_arcsec,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                meta=meta,
            )

            file_path = out_path / f"lens_{class_name}_{idx:04d}.fits"
            # FITS standard stores float32 as big-endian ('>f4') on disk automatically
            hdu = fits.PrimaryHDU(data=image, header=header)
            hdu.writeto(file_path, overwrite=True)
            created_files.append(file_path)

    return created_files


def main() -> None:
    """CLI entry point to synthesize a reproducible local FITS dataset."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Euclid-like strong lensing FITS cutouts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Destination directory for generated .fits files.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Number of synthetic FITS cutouts per class.",
    )
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    args = parser.parse_args()

    files = generate_synthetic_dataset(
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    print(f"Generated {len(files)} synthetic FITS cutouts in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
