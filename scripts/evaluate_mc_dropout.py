"""Evaluate a saved AstroLens model on the shared seed-42 test split."""

import argparse
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
    """Collect one row per image and verify the deterministic confusion matrix."""
    model.eval()
    torch.manual_seed(42)
    rows = []
    matrix = [[0 for _ in classes] for _ in classes]

    for images, targets in loader:
        probabilities = model(images).softmax(dim=-1)
        confidences, predictions = probabilities.max(dim=-1)

        samples = sample_probabilities(model, images, passes=20)
        mc_probabilities = samples.mean(dim=0)
        mc_predictions = mc_probabilities.argmax(dim=-1)
        entropy = summarize_entropy(samples)

        for i, target in enumerate(targets.tolist()):
            predicted = predictions[i].item()
            matrix[target][predicted] += 1

            # With shuffle=False, the loader follows the order in the test manifest.
            entry = entries[len(rows)]
            row = {
                "path": entry["path"],
                "true_class": classes[target],
                "det_prediction": classes[predicted],
                "det_correct": int(predicted == target),
                "det_confidence": confidences[i].item(),
                "mc_prediction": classes[mc_predictions[i].item()],
                "predictive_entropy_nats": entropy.predictive_entropy[i].item(),
                "expected_entropy_nats": entropy.expected_entropy[i].item(),
                "mutual_information_nats": entropy.mutual_information[i].item(),
            }
            for k, name in enumerate(classes):
                row[f"det_p_{name}"] = probabilities[i, k].item()
                row[f"mc_p_{name}"] = mc_probabilities[i, k].item()
            rows.append(row)

    if len(rows) != len(entries) or matrix != expected_matrix:
        raise ValueError("Per-image predictions do not reproduce the verified test result.")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "--arch",
        dest="model_name",
        choices=("resnet", "lens_vit"),
        default="resnet",
        help="Model to evaluate (default: resnet)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Benchmark run directory containing report.json and model checkpoints",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the FITS images listed in the run report",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    report_path = args.run_dir / "report.json"
    reference_path = Path(__file__).resolve().parents[1] / "docs/benchmarks/reference42.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    classes = report["classes"]
    entries = report["split"]["test"]
    if entries != reference["split"]["test"] or classes != reference["classes"]:
        raise ValueError("Test split or class order differs from the committed reference.")

    paths = [args.data_dir / entry["path"] for entry in entries]
    if len(paths) != 30 or len(set(paths)) != 30:
        raise ValueError("Expected 30 distinct test paths in the seed-42 reference.")

    for path, entry in zip(paths, entries, strict=True):
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"FITS checksum mismatch: {path}")

    dataset = FitsLensDataset(file_paths=paths, transform=ArcsinhStretch(beta=10.0))
    loader = DataLoader(dataset, batch_size=report["batch_size"], shuffle=False)

    model = build_model(args.model_name)
    checkpoint_path = args.run_dir / f"{args.model_name}.pt"
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    metrics = evaluate(model, loader)

    matrix = metrics["confusion_true_by_predicted"]
    expected = report["models"][args.model_name]["test"]["confusion_true_by_predicted"]
    correct = sum(row[i] for i, row in enumerate(matrix))

    print(f"Model: {args.model_name}")
    print(f"Verified FITS files: {len(paths)}")
    print(f"Deterministic accuracy: {correct}/{len(dataset)}")
    print(f"Columns (predicted): {', '.join(classes)}")
    for label, row in zip(classes, matrix, strict=True):
        print(f"True {label:<12} {row}")

    if matrix != expected:
        raise ValueError(f"Checkpoint predictions differ from the run report: {expected}")

    rows = collect_predictions(model, loader, classes, entries, expected)
    csv_path = args.run_dir / f"{args.model_name}_mc20_seed42_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} test predictions to {csv_path}")


if __name__ == "__main__":
    main()
