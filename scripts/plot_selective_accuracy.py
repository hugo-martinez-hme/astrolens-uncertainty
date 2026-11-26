"""Plot selective accuracy from the seed-42 WSL AstroResNet predictions."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Per-image prediction CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output figure path")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    total = len(rows)
    if total != 30 or len({row["path"] for row in rows}) != 30:
        raise ValueError("Expected 30 distinct test images for this evaluation.")

    # For each score, sort the images to send for review first.
    criteria = [
        ("Deterministic confidence", "det_confidence", False, "o"),
        ("MC predictive entropy", "predictive_entropy_nats", True, "s"),
        ("MC mutual information", "mutual_information_nats", True, "^"),
    ]
    correct_total = sum(int(row["det_correct"]) for row in rows)
    coverages = [kept / total for kept in range(1, total + 1)]
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, column, descending, marker in criteria:
        review_order = sorted(rows, key=lambda row: float(row[column]), reverse=descending)
        accuracies = []
        for kept in range(1, total + 1):
            accepted = review_order[total - kept :]
            correct = sum(int(row["det_correct"]) for row in accepted)
            accuracies.append(correct / kept)

            if kept in (24, 30):
                print(f"{label}: kept {kept}/{total}, correct {correct}/{kept}")

        ax.plot(coverages, accuracies, label=label, marker=marker, markersize=3)

    ax.axhline(
        correct_total / total,
        color="0.4",
        linestyle="--",
        label="Random selection (expected)",
    )
    ax.axvline(0.8, color="0.75", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Coverage (fraction of test images accepted)",
        ylabel="Accuracy on accepted images",
        title="AstroResNet on 30 synthetic lenses · WSL run, seed 42",
        xlim=(0, 1.02),
        ylim=(-0.03, 1.04),
    )
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()

    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
