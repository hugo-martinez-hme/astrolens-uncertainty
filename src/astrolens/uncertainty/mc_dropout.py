"""Monte Carlo dropout inference for AstroLens classifiers."""

import torch

from astrolens.models.base import BayesianModule


@torch.inference_mode()
def sample_probabilities(
    model: BayesianModule,
    batch: torch.Tensor,
    passes: int = 20,
) -> torch.Tensor:
    """Sample class probabilities and leave the model in evaluation mode.

    Args:
        model: AstroLens classifier with selective dropout control.
        batch: Input images of shape (B, C, H, W).
        passes: Number of stochastic forward passes.

    Returns:
        Class probabilities of shape (T, B, K), where T is the number
        of passes, B is the batch size and K is the number of classes.
    """
    if passes < 1:
        raise ValueError("passes must be at least 1")

    model.set_mc_dropout(True)
    try:
        samples = [model(batch).softmax(dim=-1) for _ in range(passes)]
        return torch.stack(samples)
    finally:
        model.set_mc_dropout(False)
