"""Entropy decomposition for stochastic categorical predictions."""

from typing import NamedTuple

import torch


class EntropySummary(NamedTuple):
    predictive_entropy: torch.Tensor
    expected_entropy: torch.Tensor
    mutual_information: torch.Tensor


def summarize_entropy(samples: torch.Tensor) -> EntropySummary:
    """Summarize normalized probabilities (passes, batch, classes) in nats per image."""
    if samples.ndim != 3 or samples.shape[0] == 0 or samples.shape[-1] == 0:
        raise ValueError("expected shape (passes, batch, classes) with at least one pass and class")
    if not samples.is_floating_point():
        raise TypeError("samples must contain floating-point probabilities")
    if samples.dtype in (torch.float16, torch.bfloat16):
        samples = samples.float()

    predictive = torch.special.entr(samples.mean(dim=0)).sum(dim=-1)
    expected = torch.special.entr(samples).sum(dim=-1).mean(dim=0)
    return EntropySummary(predictive, expected, predictive - expected)
