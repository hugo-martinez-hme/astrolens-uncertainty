"""Unit and architectural invariant tests for Module 2: AstroResNet vs. LensViT."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from astrolens.models import AstroResNet, LensViT, build_model


def test_iso_parameter_budget_parity() -> None:
    """Verify AstroResNet and LensViT match within 1% parameter budget (~175k params)."""
    resnet = build_model("resnet", in_channels=1, num_classes=3, dropout_p=0.1)
    vit = build_model("lens_vit", in_channels=1, num_classes=3, dropout_p=0.1)

    n_resnet = resnet.count_parameters()
    n_vit = vit.count_parameters()

    assert n_resnet == 174_515
    assert n_vit == 175_619

    relative_diff = abs(n_resnet - n_vit) / max(n_resnet, n_vit)
    assert relative_diff < 0.01, (
        f"Iso-parameter budget violated: ResNet={n_resnet}, LensViT={n_vit} "
        f"(diff={relative_diff:.2%})."
    )


@pytest.mark.parametrize("arch", ["resnet", "lens_vit"])
def test_forward_shapes_and_full_gradient_flow(arch: str) -> None:
    """Verify output shapes, finite logits, and non-zero gradients for every parameter."""
    torch.manual_seed(42)
    model = build_model(arch, in_channels=1, num_classes=3, dropout_p=0.1)
    model.train()

    x = torch.randn(4, 1, 64, 64)
    logits = model(x)

    assert logits.shape == (4, 3)
    assert torch.all(torch.isfinite(logits))

    loss = logits.pow(2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient on '{name}' in {arch}."
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradient on '{name}'."


@pytest.mark.parametrize("arch", ["resnet", "lens_vit"])
def test_deterministic_eval_vs_stochastic_mc_dropout(arch: str) -> None:
    """Verify eval() is deterministic while set_mc_dropout(True) yields stochastic samples."""
    torch.manual_seed(7)
    model = build_model(arch, in_channels=1, num_classes=3, dropout_p=0.2)
    x = torch.randn(2, 1, 64, 64)

    # 1. Standard deterministic evaluation
    model.set_mc_dropout(False)
    with torch.no_grad():
        out_det_1 = model(x)
        out_det_2 = model(x)
    assert torch.equal(out_det_1, out_det_2)

    # 2. Stochastic MC Dropout inference
    model.set_mc_dropout(True)
    with torch.no_grad():
        samples = torch.stack([model(x) for _ in range(10)], dim=0)  # (T, B, K)

    sample_std = samples.std(dim=0)
    assert torch.all(sample_std > 0.0), f"MC Dropout produced zero variance in {arch}."


def test_batchnorm_frozen_during_mc_dropout_single_sample() -> None:
    """Ensure BatchNorm2d stays frozen during MC Dropout (no stats drift and batch_size=1 safe)."""
    torch.manual_seed(11)
    model = AstroResNet(in_channels=1, num_classes=3, dropout_p=0.2)

    # Simulate one training step so running stats move away from initialization
    model.train()
    _ = model(torch.randn(8, 1, 64, 64))

    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bn_layers) > 0
    saved_means = [bn.running_mean.clone() for bn in bn_layers if bn.running_mean is not None]
    saved_vars = [bn.running_var.clone() for bn in bn_layers if bn.running_var is not None]

    # Activate MC Dropout and run single-sample inference (batch_size=1 crashes unfrozen BN)
    model.set_mc_dropout(True)
    for bn in bn_layers:
        assert bn.training is False

    x_single = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        logits = model(x_single)
    assert logits.shape == (1, 3)

    # Verify running statistics remained bit-exact untouched
    for bn, orig_mean, orig_var in zip(bn_layers, saved_means, saved_vars, strict=True):
        assert bn.running_mean is not None and torch.equal(bn.running_mean, orig_mean)
        assert bn.running_var is not None and torch.equal(bn.running_var, orig_var)


def test_lens_vit_attention_row_stochasticity() -> None:
    """Verify LensViT attention weights form valid probability distributions over all 65 tokens."""
    torch.manual_seed(3)
    model = LensViT(in_channels=1, image_size=64, patch_size=8, num_classes=3, dropout_p=0.0)
    model.eval()

    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        logits, attn = model(x, return_last_attention=True)

    assert logits.shape == (2, 3)
    # 64 spatial patches + 1 [CLS] token = 65 tokens across 4 heads
    assert attn.shape == (2, 4, 65, 65)
    assert torch.all(attn >= 0.0)
    row_sums = attn.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5)
