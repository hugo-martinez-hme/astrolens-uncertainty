"""Base interface for Bayesian-ready neural architectures in AstroLens."""

from __future__ import annotations

import torch.nn as nn

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


class BayesianModule(nn.Module):
    """Abstract base class providing parameter accounting and safe MC Dropout control."""

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total number of model parameters.

        Args:
            trainable_only: If True, counts only parameters with requires_grad=True.

        Returns:
            Exact integer count of parameters.
        """
        return sum(p.numel() for p in self.parameters() if (not trainable_only or p.requires_grad))

    def set_mc_dropout(self, enabled: bool) -> None:
        """Switch model to inference mode with optional selective Monte Carlo Dropout.

        Sets the entire module to ``eval()`` first so that normalization layers
        (e.g., ``BatchNorm2d``) strictly preserve their learned running statistics,
        then selectively enables ``train()`` exclusively on dropout layers when
        ``enabled=True``.

        Args:
            enabled: If True, activates stochastic dropout sampling during inference.
        """
        self.eval()
        if enabled:
            for module in self.modules():
                if isinstance(module, _DROPOUT_TYPES):
                    module.train()
