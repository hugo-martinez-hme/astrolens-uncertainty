"""Iso-parameter neural architectures for controlled scientific ablation."""

from astrolens.models.base import BayesianModule
from astrolens.models.lens_vit import LensViT
from astrolens.models.resnet import AstroResNet


def build_model(
    arch: str,
    in_channels: int = 1,
    num_classes: int = 3,
    dropout_p: float = 0.1,
) -> BayesianModule:
    """Factory instantiating iso-parameter benchmark architectures by name.

    Args:
        arch: Architecture identifier ('resnet' or 'lens_vit').
        in_channels: Number of photometric input channels.
        num_classes: Number of morphological classes.
        dropout_p: Variational dropout probability.

    Returns:
        Instantiated BayesianModule ready for training or MC Dropout inference.

    Raises:
        ValueError: If arch is not recognized.
    """
    key = arch.lower().strip()
    if key == "resnet":
        return AstroResNet(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )
    if key in ("lens_vit", "vit"):
        return LensViT(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )
    raise ValueError(f"Unknown architecture '{arch}'. Expected 'resnet' or 'lens_vit'.")


__all__ = [
    "AstroResNet",
    "BayesianModule",
    "LensViT",
    "build_model",
]
