from itertools import combinations

import numpy as np
import pytest
from astropy.io import fits

from astrolens.benchmark import stratified_split


def test_paired_split_is_reproducible_disjoint_and_balanced(tmp_path):
    for label in range(3):
        for i in range(10):
            header = fits.Header({"CLASS_ID": label})
            fits.writeto(tmp_path / f"{label}_{i}.fits", np.zeros((2, 2)), header)
    split = stratified_split(tmp_path, seed=42)
    assert split == stratified_split(tmp_path, seed=42)
    assert split != stratified_split(tmp_path, seed=10)
    for left, right in combinations(split.values(), 2):
        assert set(left).isdisjoint(right)
    assert set().union(*split.values()) == set(tmp_path.glob("*.fits"))
    for key, per_class in (("train", 6), ("validation", 2), ("test", 2)):
        labels = [fits.getval(path, "CLASS_ID") for path in split[key]]
        assert [labels.count(i) for i in range(3)] == [per_class] * 3


def test_split_rejects_missing_classes(tmp_path):
    with pytest.raises(ValueError, match="five independent images"):
        stratified_split(tmp_path, seed=42)
