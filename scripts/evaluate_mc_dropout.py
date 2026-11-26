"""Evaluate the saved AstroResNet checkpoint on the reference test split."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from astrolens.benchmark import evaluate
from astrolens.data.dataset import FitsLensDataset
from astrolens.data.transforms import ArcsinhStretch
from astrolens.models import build_model


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run_dir = Path("results/baseline_wsl_42")
    data_dir = Path("data/synthetic")
    report = json.loads((run_dir / "report.json").read_text())
    reference = json.loads(Path("docs/benchmarks/reference42.json").read_text())

    entries = report["split"]["test"]
    if entries != reference["split"]["test"] or report["classes"] != reference["classes"]:
        raise ValueError("Test split or class order differs from the committed reference.")

    paths = [data_dir / entry["path"] for entry in entries]
    if len(paths) != 30 or len(set(paths)) != 30:
        raise ValueError("Expected 30 distinct test paths in the seed-42 reference.")

    for path, entry in zip(paths, entries, strict=True):
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"FITS checksum mismatch: {path}")

    dataset = FitsLensDataset(file_paths=paths, transform=ArcsinhStretch(beta=10.0))
    loader = DataLoader(dataset, batch_size=report["batch_size"], shuffle=False)

    model = build_model("resnet")
    weights = torch.load(run_dir / "resnet.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    metrics = evaluate(model, loader)

    matrix = metrics["confusion_true_by_predicted"]
    expected = report["models"]["resnet"]["test"]["confusion_true_by_predicted"]
    correct = sum(row[i] for i, row in enumerate(matrix))

    print(f"Verified FITS files: {len(paths)}")
    print(f"Deterministic accuracy: {correct}/{len(dataset)}")
    print(f"Columns (predicted): {', '.join(report['classes'])}")
    for label, row in zip(report["classes"], matrix, strict=True):
        print(f"True {label:<12} {row}")

    if matrix != expected:
        raise ValueError(f"Checkpoint predictions differ from the WSL run report: {expected}")


if __name__ == "__main__":
    main()
