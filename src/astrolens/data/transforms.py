"""Photometric dynamic-range compression and exact D4 lattice symmetry operators."""

from __future__ import annotations

import torch
import torch.nn as nn


class ArcsinhStretch(nn.Module):
    r"""Lupton inverse hyperbolic sine photometric stretch for high-dynamic-range fields.

    Maps physical sky-subtracted flux :math:`x` via:
    .. math::
        f(x; \beta) = \operatorname{arcsinh}\!\left(\frac{x - \mu_{\text{sky}}}{\beta}\right)

    Asymptotic regimes:
        - Low SNR (:math:`|x - \mu_{\text{sky}}| \ll \beta`): linear
          :math:`f(x) \approx (x - \mu_{\text{sky}}) / \beta`, smoothly handling negative
          sky fluctuations without NaN singularities.
        - High SNR (:math:`x - \mu_{\text{sky}} \gg \beta`): logarithmic
          :math:`f(x) \approx \ln(2(x - \mu_{\text{sky}}) / \beta)`, preventing bright
          galaxy cores from saturating activations or gradients.
    """

    def __init__(self, beta: float = 10.0, sky_level: float = 0.0) -> None:
        """Initialize the photometric stretch operator.

        Args:
            beta: Strictly positive softening scale (typically 1-3x sky RMS in ADU).
            sky_level: Optional residual sky baseline to subtract prior to scaling.

        Raises:
            ValueError: If beta is not strictly positive or finite.
        """
        super().__init__()
        if not (beta > 0.0 and torch.isfinite(torch.tensor(beta))):
            raise ValueError(f"Softening parameter beta must be finite and > 0, got {beta}.")

        self.register_buffer("beta", torch.tensor(float(beta), dtype=torch.float32))
        self.register_buffer("sky_level", torch.tensor(float(sky_level), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply vectorized arcsinh compression to a tensor of shape (..., H, W)."""
        return torch.asinh((x - self.sky_level) / self.beta)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Invert the arcsinh transform back to linear physical flux units."""
        return torch.sinh(y) * self.beta + self.sky_level


class DihedralD4Transform(nn.Module):
    r"""Exact unitary action of the Dihedral Group :math:`D_4` on square 2D pixel lattices.

    Presentation:
    .. math::
        D_4 = \langle r, s \mid r^4 = e,\; s^2 = e,\; s r s = r^{-1} \rangle

    Element indexing :math:`k \in \{0, \dots, 7\}`:
        - :math:`k \in \{0, 1, 2, 3\}`: Counter-clockwise rotations :math:`r^k` by
          :math:`90^\circ \cdot k`.
        - :math:`k \in \{4, 5, 6, 7\}`: Horizontal flip :math:`s` (along width axis)
          followed by counter-clockwise rotation :math:`r^{k - 4}`, i.e., :math:`r^{k - 4} s`.

    Because every group action is an exact coordinate permutation on :math:`\mathbb{Z}^2`,
    it introduces zero interpolation blur and strictly conserves :math:`\ell_2` norm and
    pixel-level Poisson-Gaussian noise statistics.
    """

    GROUP_ORDER: int = 8

    def __init__(self, k: int | None = None, p: float = 1.0) -> None:
        """Initialize the D4 symmetry transform.

        Args:
            k: Optional deterministic group element index in {0, ..., 7}. If None,
                samples uniformly from D4 at every forward call with probability p.
            p: Probability of applying a random D4 element when k is None.

        Raises:
            ValueError: If k is out of range or p is outside [0, 1].
        """
        super().__init__()
        if k is not None and not (0 <= k < self.GROUP_ORDER):
            raise ValueError(f"Deterministic group index k must be in [0, 7], got {k}.")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability p must be in [0, 1], got {p}.")

        self.k = k
        self.p = p

    @staticmethod
    def _validate_lattice(x: torch.Tensor) -> None:
        """Verify tensor has at least 2 spatial dimensions and forms a square lattice."""
        if x.ndim < 2:
            raise ValueError(f"Expected tensor with shape (..., H, W), got shape {tuple(x.shape)}.")
        if x.shape[-2] != x.shape[-1]:
            raise ValueError(
                f"D4 lattice closure requires H == W, got H={x.shape[-2]}, W={x.shape[-1]}."
            )

    @classmethod
    def apply_element(cls, x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply deterministic group element k in {0, ..., 7} to tensor (..., H, W)."""
        if not 0 <= k < cls.GROUP_ORDER:
            raise ValueError(f"Group index k must be in [0, 7], got {k}.")
        cls._validate_lattice(x)

        out = torch.flip(x, dims=(-1,)) if k >= 4 else x
        rot_steps = k % 4
        if rot_steps > 0:
            out = torch.rot90(out, k=rot_steps, dims=(-2, -1))
        return out

    @classmethod
    def inverse_index(cls, k: int) -> int:
        """Return the exact algebraic inverse index k^{-1} in D4.

        For pure rotations r^m (k in 0..3): (r^m)^{-1} = r^{(4 - m) % 4}.
        For reflections r^m s (k in 4..7): every reflection is an involution
        (r^m s)^2 = r^m (s r^m s) = r^m r^{-m} = e, hence (r^m s)^{-1} = r^m s.
        """
        if not 0 <= k < cls.GROUP_ORDER:
            raise ValueError(f"Group index k must be in [0, 7], got {k}.")
        return (4 - k) % 4 if k < 4 else k

    @classmethod
    def invert_element(cls, x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply the exact inverse group element g_k^{-1} to recover the original orientation."""
        return cls.apply_element(x, cls.inverse_index(k))

    @classmethod
    def orbit_expand(cls, x: torch.Tensor) -> torch.Tensor:
        """Expand a single image (C, H, W) or batch (B, C, H, W) across all 8 D4 states.

        Returns:
            Tensor of shape (8, C, H, W) for 3D inputs or (B, 8, C, H, W) for 4D inputs.
        """
        cls._validate_lattice(x)
        views = [cls.apply_element(x, idx) for idx in range(cls.GROUP_ORDER)]
        stack_dim = 1 if x.ndim == 4 else 0
        return torch.stack(views, dim=stack_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply deterministic element k (if set) or a random uniform D4 symmetry."""
        if self.k is not None:
            return self.apply_element(x, self.k)
        if self.p < 1.0 and float(torch.rand(1).item()) > self.p:
            return x
        random_k = int(torch.randint(0, self.GROUP_ORDER, (1,)).item())
        return self.apply_element(x, random_k)