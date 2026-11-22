"""Tests for stochastic inference with frozen normalization statistics."""

import torch
from torch import nn

from astrolens.models import build_model
from astrolens.uncertainty.mc_dropout import sample_probabilities


def test_mc_dropout_samples_preserve_batchnorm_state():
    torch.manual_seed(42)
    model = build_model("resnet")
    model.eval()
    batch = torch.randn(2, 1, 64, 64)

    # Keep independent copies so in-place updates would be detected.
    batchnorms = [layer for layer in model.modules() if isinstance(layer, nn.BatchNorm2d)]
    assert batchnorms
    before = [
        (layer.running_mean.clone(), layer.running_var.clone(), layer.num_batches_tracked.clone())
        for layer in batchnorms
    ]

    samples = sample_probabilities(model, batch, passes=4)

    assert samples.shape == (4, 2, 3)
    torch.testing.assert_close(samples.sum(dim=-1), torch.ones(4, 2))
    assert (samples[1:] != samples[0]).any()
    assert all(not layer.training for layer in model.modules())

    for layer, (mean, variance, count) in zip(batchnorms, before, strict=True):
        assert torch.equal(layer.running_mean, mean)
        assert torch.equal(layer.running_var, variance)
        assert torch.equal(layer.num_batches_tracked, count)
