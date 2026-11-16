"""Compact Bayesian-ready ResNet with spatial Dropout2d for strong gravitational lensing."""

from __future__ import annotations

import torch
import torch.nn as nn

from astrolens.models.base import BayesianModule


class ResidualBlock(nn.Module):
    r"""Two-layer convolutional residual block with structured spatial Dropout2d.

    Computes
    :math:`\mathbf{y} = \operatorname{ReLU}(\mathcal{F}(\mathbf{x}) + \mathcal{S}(\mathbf{x}))`,
    placing ``Dropout2d`` strictly inside the residual branch :math:`\mathcal{F}` so the
    identity skip connection :math:`\mathcal{S}` preserves clean gradient propagation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_p: float = 0.1,
    ) -> None:
        """Initialize the residual block.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            stride: Spatial downsampling stride applied in the first convolution.
            dropout_p: Channel-wise spatial dropout probability for variational inference.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.spatial_drop = nn.Dropout2d(p=dropout_p)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual transformation to tensor of shape (B, C_in, H, W)."""
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.spatial_drop(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class AstroResNet(BayesianModule):
    """Compact convolutional ResNet tailored for 64x64 astronomical cutouts (~174.5k params).

    Architecture (default ``base_channels=16``):
        - Stem: 3x3 Conv (stride 2) + BN + ReLU -> (16, 32, 32)
        - Stage 1: 2x ResidualBlock (16 -> 16, stride 1) -> (16, 32, 32)
        - Stage 2: 2x ResidualBlock (16 -> 32, stride 2) -> (32, 16, 16)
        - Stage 3: 2x ResidualBlock (32 -> 64, stride 2) -> (64, 8, 8)
        - Head: Global Average Pooling + Dropout + Linear(64 -> num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 16,
        dropout_p: float = 0.1,
    ) -> None:
        """Initialize AstroResNet.

        Args:
            in_channels: Number of input photometric bands/channels.
            num_classes: Number of target morphological classes.
            base_channels: Width multiplier for stage 1 (stages scale as c, 2c, 4c).
            dropout_p: Dropout probability used both during training and MC inference.

        Raises:
            ValueError: If dropout_p is outside [0, 1) or base_channels <= 0.
        """
        super().__init__()
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}.")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be strictly positive, got {base_channels}.")

        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(c, c, num_blocks=2, stride=1, dropout_p=dropout_p)
        self.stage2 = self._make_stage(c, 2 * c, num_blocks=2, stride=2, dropout_p=dropout_p)
        self.stage3 = self._make_stage(2 * c, 4 * c, num_blocks=2, stride=2, dropout_p=dropout_p)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(4 * c, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout_p: float,
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride=stride, dropout_p=dropout_p)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, dropout_p=dropout_p))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Apply Kaiming normal initialization for ReLU convolutional layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass mapping (B, C, H, W) tensor to unnormalized class logits (B, K)."""
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out).flatten(1)
        out = self.head_drop(out)
        return self.fc(out)
