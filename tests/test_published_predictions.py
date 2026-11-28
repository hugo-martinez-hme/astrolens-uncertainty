"""Regression checks for the published seed-42 prediction tables (no retraining)."""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
CRITERIA = (
    ("det_confidence", False),
    ("predictive_entropy_nats", True),
    ("mutual_information_nats", True),
)


def read_rows(arch):
    with (BENCHMARKS / f"{arch}_mc20_seed42_test.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_report(name):
    return json.loads((BENCHMARKS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("arch", "expected_matrix", "expected_errors"),
    [
        ("resnet", [[0, 10, 0], [0, 10, 0], [0, 0, 10]], (4, 3, 1)),
        ("lens_vit", [[0, 10, 0], [0, 9, 1], [0, 0, 10]], (5, 5, 5)),
    ],
)
def test_published_predictions(arch, expected_matrix, expected_errors):
    reference = read_report("reference42.json")
    classes = reference["classes"]
    rows = read_rows(arch)
    paths = [row["path"] for row in rows]
    assert len(rows) == len(set(paths)) == 30
    assert paths == [entry["path"] for entry in reference["split"]["test"]]
    assert Counter(row["true_class"] for row in rows) == dict.fromkeys(classes, 10)

    matrix = [[0 for _ in classes] for _ in classes]
    for row in rows:
        true = row["true_class"]
        predicted = row["det_prediction"]
        assert Path(row["path"]).name.startswith(f"lens_{true}_")
        assert int(row["det_correct"]) == int(true == predicted)
        matrix[classes.index(true)][classes.index(predicted)] += 1

        for mode, prediction in (("det", predicted), ("mc", row["mc_prediction"])):
            probs = [float(row[f"{mode}_p_{name}"]) for name in classes]
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
            assert sum(probs) == pytest.approx(1.0, abs=1e-6)
            assert classes[probs.index(max(probs))] == prediction
            if mode == "det":
                assert float(row["det_confidence"]) == pytest.approx(max(probs), abs=1e-6)
            else:
                entropy = -sum(p * math.log(p) for p in probs if p > 0)
                assert float(row["predictive_entropy_nats"]) == pytest.approx(entropy, abs=1e-6)

        predictive = float(row["predictive_entropy_nats"])
        expected = float(row["expected_entropy_nats"])
        mutual = float(row["mutual_information_nats"])
        assert 0 <= expected <= math.log(len(classes)) + 1e-6
        assert mutual >= -1e-6
        assert mutual == pytest.approx(predictive - expected, abs=1e-6)

    assert matrix == expected_matrix
    for (column, descending), count in zip(CRITERIA, expected_errors, strict=True):
        review = sorted(rows, key=lambda row: float(row[column]), reverse=descending)[:6]
        assert sum(1 - int(row["det_correct"]) for row in review) == count
