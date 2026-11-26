"""Evaluate the saved AstroResNet checkpoint on the reference test split."""

import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from astrolens.benchmark import evaluate
from astrolens.data.dataset import FitsLensDataset
from astrolens.data.transforms import ArcsinhStretch
from astrolens.models import build_model
from astrolens.uncertainty.entropy import summarize_entropy
from astrolens.uncertainty.mc_dropout import sample_probabilities


@torch.inference_mode()
def collect_predictions(model, loader, classes, entries, expected_matrix):
    """Collect one row per test image and verify the deterministic predictions."""
    model.eval()
    torch.manual_seed(42)
    rows = []
    matrix = [[0 for _ in classes] for _ in classes]

    for images, targets in loader:
        probabilities = model(images).softmax(dim=-1)
        confidences, predictions = probabilities.max(dim=-1)

        samples = sample_probabilities(model, images, passes=20)
        mc_predictions = samples.mean(dim=0).argmax(dim=-1)
        entropy = summarize_entropy(samples)

        for i, target in enumerate(targets.tolist()):
            predicted = predictions[i].item()
            matrix[target][predicted] += 1
            rows.append(
                {
                    "path": entries[len(rows)]["path"],
                    "true_class": classes[target],
                    "det_prediction": classes[predicted],
                    "det_correct": int(predicted == target),
                    "det_confidence": confidences[i].item(),
                    "mc_prediction": classes[mc_predictions[i].item()],
                    "predictive_entropy_nats": entropy.predictive_entropy[i].item(),
                    "expected_entropy_nats": entropy.expected_entropy[i].item(),
                    "mutual_information_nats": entropy.mutual_information[i].item(),
                }
            )

    if len(rows) != len(entries) or matrix != expected_matrix:
        raise ValueError("Per-image predictions do not reproduce the verified test result.")

    return rows


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

    rows = collect_predictions(model, loader, report["classes"], entries, expected)
    csv_path = run_dir / "resnet_mc20_seed42_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} test predictions to {csv_path}")


if __name__ == "__main__":
    main()
