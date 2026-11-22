"""Paired CPU benchmark on independently simulated, sky-subtracted FITS images."""

import argparse
import hashlib
import json
import platform
import random
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import torch
from astropy.io import fits
from torch import nn
from torch.utils.data import DataLoader

from astrolens.data.dataset import FitsLensDataset
from astrolens.data.synthetic_fits import CLASS_NAMES
from astrolens.data.transforms import ArcsinhStretch
from astrolens.models import build_model


def stratified_split(root: Path, seed: int) -> dict[str, list[Path]]:
    """Reserve one fifth per class for validation and another fifth for testing."""
    groups = defaultdict(list)
    for path in sorted(root.rglob("*.fits")):
        label = fits.getval(path, "CLASS_ID")
        if label not in range(len(CLASS_NAMES)):
            raise ValueError(f"Unexpected CLASS_ID {label} in {path}")
        groups[label].append(path)
    rng = random.Random(seed)
    split = {"train": [], "validation": [], "test": []}
    for label, name in enumerate(CLASS_NAMES):
        paths = groups[label]
        if len(paths) < 5:
            raise ValueError(f"Need at least five independent images for class {name}")
        rng.shuffle(paths)
        n = len(paths) // 5
        split["test"].extend(paths[:n])
        split["validation"].extend(paths[n : 2 * n])
        split["train"].extend(paths[2 * n :])
    return split


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    loss = 0.0
    classes = len(CLASS_NAMES)
    counts = torch.zeros(classes, classes, dtype=torch.long)
    for images, targets in loader:
        logits = model(images)
        loss += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        pairs = targets * classes + logits.argmax(1)
        counts += torch.bincount(pairs, minlength=classes**2).reshape(classes, classes)
    correct = counts.diag()
    return {
        "nll": loss / len(loader.dataset),
        "accuracy": correct.sum().item() / len(loader.dataset),
        "macro_f1": (2 * correct / (counts.sum(0) + counts.sum(1)).clamp_min(1)).mean().item(),
        "confusion_true_by_predicted": counts.tolist(),
    }


def fit(arch, split, output, seed, epochs, batch_size):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    datasets = {
        key: FitsLensDataset(file_paths=paths, transform=ArcsinhStretch(beta=10.0))
        for key, paths in split.items()
    }
    train = DataLoader(datasets["train"], batch_size, shuffle=True, generator=generator)
    validation = DataLoader(datasets["validation"], batch_size)
    model = build_model(arch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    checkpoint = output / f"{arch}.pt"
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train:
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(images), labels).backward()
            optimizer.step()
        metrics = evaluate(model, validation)
        history.append(metrics)
        if metrics["nll"] < best_loss:
            best_epoch = epoch
            best_loss = metrics["nll"]
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    return {
        "parameters": model.count_parameters(),
        "best_epoch": best_epoch,
        "validation_by_epoch": history,
        "test": evaluate(model, DataLoader(datasets["test"], batch_size)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("epochs and batch-size must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    split = stratified_split(args.data, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "classes": CLASS_NAMES,
        "python": platform.python_version(),
        "versions": {k: version(k) for k in ("torch", "astropy", "numpy")},
        "split": {
            key: [
                {
                    "path": p.relative_to(args.data).as_posix(),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in paths
            ]
            for key, paths in split.items()
        },
        "models": {},
    }
    for arch in ("resnet", "lens_vit"):
        result = fit(arch, split, args.output, args.seed, args.epochs, args.batch_size)
        report["models"][arch] = result
        print(arch, result["test"], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
