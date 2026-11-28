"""Plot selective accuracy on the seed-42 synthetic lens test split."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

CRITERIA = (
    ("Deterministic confidence", "det_confidence", False, "o"),
    ("MC predictive entropy", "predictive_entropy_nats", True, "s"),
    ("MC mutual information", "mutual_information_nats", True, "^"),
)


def read_predictions(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 30 or len({row["path"] for row in rows}) != 30:
        raise ValueError(f"Expected 30 distinct test images in {path}")
    return rows


def plot_panel(ax, name, rows):
    total = len(rows)
    coverages = [kept / total for kept in range(1, total + 1)]

    for label, column, descending, marker in CRITERIA:
        # Stable sorting preserves manifest order for ties in the review queue.
        review_order = sorted(rows, key=lambda row: float(row[column]), reverse=descending)
        accuracies = []
        correct = 0
        for kept, row in enumerate(reversed(review_order), start=1):
            correct += int(row["det_correct"])
            accuracies.append(correct / kept)
            if kept in (24, 30):
                print(f"{name} | {label}: kept {kept}/{total}, correct {correct}/{kept}")

        ax.plot(coverages, accuracies, label=label, marker=marker, markersize=3)

    baseline = sum(int(row["det_correct"]) for row in rows) / total
    ax.axhline(baseline, color="0.4", linestyle="--", label="Random selection (expected)")
    ax.axvline(0.8, color="0.75", linestyle=":", linewidth=1)
    ax.set(title=name, xlabel="Coverage", xlim=(0, 1.02), ylim=(-0.03, 1.04))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="AstroResNet prediction CSV")
    parser.add_argument("--vit-csv", type=Path, help="Paired LensViT prediction CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output figure path")
    args = parser.parse_args()

    resnet = read_predictions(args.csv)
    panels = [("AstroResNet", resnet)]
    if args.vit_csv is not None:
        vit = read_predictions(args.vit_csv)
        for cnn_row, vit_row in zip(resnet, vit, strict=True):
            for key in ("path", "true_class"):
                if cnn_row[key] != vit_row[key]:
                    raise ValueError(f"Paired CSVs differ in {key} at {cnn_row['path']}")
        panels.append(("LensViT", vit))

    width = 6 * len(panels) + 1
    fig, axes = plt.subplots(
        1, len(panels), figsize=(width, 5), sharex=True, sharey=True, squeeze=False
    )
    for ax, (name, rows) in zip(axes[0], panels, strict=True):
        plot_panel(ax, name, rows)

    axes[0, 0].set_ylabel("Accuracy on accepted images")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Selective accuracy on 30 synthetic lenses · WSL run, seed 42")
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
