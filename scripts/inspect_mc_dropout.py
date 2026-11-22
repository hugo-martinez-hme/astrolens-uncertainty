"""Inspect MC Dropout on one held-out smooth lens from the WSL baseline."""

import hashlib
import json
from pathlib import Path

import torch

from astrolens.data.dataset import FitsLensDataset
from astrolens.data.transforms import ArcsinhStretch
from astrolens.models import build_model


@torch.inference_mode()
def main() -> None:
    run_dir = Path("results/baseline_wsl_42")
    data_dir = Path("data/synthetic")
    report = json.loads((run_dir / "report.json").read_text())
    classes = report["classes"]
    test_entries = report["split"]["test"]

    dataset = FitsLensDataset(
        file_paths=[data_dir / entry["path"] for entry in test_entries],
        transform=ArcsinhStretch(beta=10.0),
    )
    smooth_id = classes.index("smooth")
    for index, entry in enumerate(test_entries):
        image, label = dataset[index]
        if label == smooth_id:
            path = data_dir / entry["path"]
            break
    else:
        raise ValueError("The test split contains no smooth lenses.")

    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"The FITS file differs from the recorded test image: {path}")

    model = build_model("resnet")
    model.load_state_dict(torch.load(run_dir / "resnet.pt", map_location="cpu", weights_only=True))
    model.eval()
    batch = image.unsqueeze(0)
    deterministic = model(batch).softmax(dim=-1)[0]

    torch.manual_seed(42)
    model.set_mc_dropout(True)
    passes = 20
    samples = torch.stack([model(batch).softmax(dim=-1)[0] for _ in range(passes)])
    model.set_mc_dropout(False)

    mean = samples.mean(dim=0)
    std = samples.std(dim=0, correction=0)
    votes = samples.argmax(dim=1).bincount(minlength=len(classes))

    print(f"File: {entry['path']}")
    print(f"True class: {classes[label]}")
    print(f"Deterministic prediction: {classes[deterministic.argmax().item()]}")
    print(f"MC mean prediction: {classes[mean.argmax().item()]}")
    print(f"{'Class':<14} {'Det.':>8} {'MC mean':>8} {'MC std':>8} {'Votes':>7}")
    for i, name in enumerate(classes):
        print(
            f"{name:<14} {deterministic[i]:8.4f} {mean[i]:8.4f}"
            f" {std[i]:8.4f} {votes[i].item():3d}/{passes}"
        )


if __name__ == "__main__":
    main()
