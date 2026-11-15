"""Generate publication-quality diagnostic figure for Module 1 physical pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from astrolens.data.synthetic_fits import CLASS_NAMES, simulate_lens_image
from astrolens.data.transforms import ArcsinhStretch, DihedralD4Transform

D4_LABELS: tuple[str, ...] = (
    r"$e$ ($0^\circ$)",
    r"$r$ ($90^\circ$)",
    r"$r^2$ ($180^\circ$)",
    r"$r^3$ ($270^\circ$)",
    r"$s$ (flip H)",
    r"$rs$ (diag)",
    r"$r^2 s$ (flip V)",
    r"$r^3 s$ (antidiag)",
)

CLASS_TITLES: dict[str, str] = {
    "smooth": "Class 0: Smooth Halo",
    "cdm_subhalo": "Class 1: CDM Subhalos",
    "anomaly": "Class 2: Rare Anomaly (Double Ring)",
}


def generate_inspection_figure(output_path: Path, beta: float = 10.0, seed: int = 42) -> None:
    """Render raw linear vs. arcsinh stretch across classes and the full 8-element D4 orbit."""
    plt.switch_backend("Agg")
    rng = np.random.default_rng(seed)
    stretch = ArcsinhStretch(beta=beta)

    samples_raw: list[torch.Tensor] = []
    samples_stretched: list[torch.Tensor] = []

    for class_id in range(len(CLASS_NAMES)):
        img_adu, _ = simulate_lens_image(class_id=class_id, rng=rng)
        tensor = torch.from_numpy(img_adu).unsqueeze(0)  # (1, 64, 64)
        samples_raw.append(tensor)
        samples_stretched.append(stretch(tensor))

    # Build 8-element D4 orbit on the CDM subhalo sample (class 1)
    d4_orbit = DihedralD4Transform.orbit_expand(samples_stretched[1])  # (8, 1, 64, 64)

    fig = plt.figure(figsize=(14, 8.6), constrained_layout=True)
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.35, 0.65], figure=fig, hspace=0.08)

    # Top block: 2 rows (Raw Linear vs. Arcsinh) x 3 columns (Classes)
    top_grid = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[0], wspace=0.08, hspace=0.1
    )
    for col, class_name in enumerate(CLASS_NAMES):
        raw_np = samples_raw[col].squeeze(0).numpy()
        str_np = samples_stretched[col].squeeze(0).numpy()

        ax_raw = fig.add_subplot(top_grid[0, col])
        im_raw = ax_raw.imshow(raw_np, origin="lower", cmap="inferno")
        ax_raw.set_title(CLASS_TITLES[class_name], fontsize=11, fontweight="bold")
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        if col == 0:
            ax_raw.set_ylabel("Raw Linear (ADU)\n[Core Saturates]", fontsize=10, fontweight="bold")
        fig.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.03)

        ax_str = fig.add_subplot(top_grid[1, col])
        im_str = ax_str.imshow(str_np, origin="lower", cmap="inferno")
        ax_str.set_xticks([])
        ax_str.set_yticks([])
        if col == 0:
            ax_str.set_ylabel(
                rf"$\operatorname{{arcsinh}}(x / \beta)$, $\beta={beta:.0f}$",
                fontsize=10,
                fontweight="bold",
            )
        fig.colorbar(im_str, ax=ax_str, fraction=0.046, pad=0.03)

    # Bottom block: 1 row x 8 columns showing the exact D4 orbit
    bot_grid = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[1], wspace=0.06)
    vmin = float(d4_orbit.min().item())
    vmax = float(d4_orbit.max().item())

    for k in range(8):
        ax_d4 = fig.add_subplot(bot_grid[0, k])
        ax_d4.imshow(
            d4_orbit[k].squeeze(0).numpy(), origin="lower", cmap="inferno", vmin=vmin, vmax=vmax
        )
        ax_d4.set_title(D4_LABELS[k], fontsize=9.5)
        ax_d4.set_xticks([])
        ax_d4.set_yticks([])
        if k == 0:
            ax_d4.set_ylabel(
                r"$D_4$ Exact Orbit" + "\n(CDM Sample)", fontsize=10, fontweight="bold"
            )

    fig.suptitle(
        "AstroLens-Uncertainty — Module 1: Photometric Stretch & Exact $D_4$ Lattice Symmetries",
        fontsize=13,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved inspection figure to '{output_path}'.")


def main() -> None:
    generate_inspection_figure(Path("docs/figures/module1_physics_pipeline.png"))


if __name__ == "__main__":
    main()
