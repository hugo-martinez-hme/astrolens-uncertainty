"""Compact Bayesian-ready Vision Transformer (LensViT) matched in parameters to AstroResNet."""

from __future__ import annotations

import torch
import torch.nn as nn

from astrolens.models.base import BayesianModule


class MultiHeadSelfAttention(nn.Module):
    r"""Explicit Scaled Dot-Product Multi-Head Self-Attention with variational dropout.

    Computes per head:
    .. math::
        \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
        = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)
          \mathbf{V}
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float = 0.1) -> None:
        """Initialize Multi-Head Self-Attention.

        Args:
            embed_dim: Total token embedding dimension D.
            num_heads: Number of parallel attention heads H (must divide embed_dim).
            dropout_p: Dropout probability on attention weights and output projection.

        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.attn_drop = nn.Dropout(p=dropout_p)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_drop = nn.Dropout(p=dropout_p)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute global self-attention over sequence (B, N, D)."""
        batch_size, seq_len, embed_dim = x.shape

        # Project and split into Q, K, V: each of shape (B, H, N, d_k)
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention scores: (B, H, N, N)
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_dropped = self.attn_drop(attn_weights)

        # Aggregate values and recombine heads: (B, N, D)
        out = (attn_dropped @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj_drop(self.proj(out))

        if return_attention:
            return out, attn_weights
        return out


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block for stable small-dataset optimization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_p=dropout_p)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, embed_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply Pre-LN attention and feed-forward residual sub-blocks."""
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights

        attn_out = self.attn(self.norm1(x), return_attention=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class LensViT(BayesianModule):
    """Compact Vision Transformer iso-parameter matched to AstroResNet (~175.6k params).

    Splits a 64x64 cutout into 8x8 patches (64 spatial tokens + 1 [CLS] token) and applies
    4 Pre-LN Transformer encoder layers with 4 heads and embedding dimension 64.
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 64,
        patch_size: int = 8,
        num_classes: int = 3,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 3.0,
        dropout_p: float = 0.1,
    ) -> None:
        """Initialize LensViT.

        Args:
            in_channels: Number of input photometric channels.
            image_size: Expected spatial height and width of square input cutouts.
            patch_size: Spatial size of each non-overlapping patch token.
            num_classes: Number of target morphological classes.
            embed_dim: Token latent dimension D.
            depth: Number of stacked Transformer encoder blocks.
            num_heads: Number of attention heads per block.
            mlp_ratio: Expansion ratio of the internal feed-forward MLP.
            dropout_p: Dropout probability used for training and MC Dropout inference.

        Raises:
            ValueError: If image_size is not divisible by patch_size or dropout_p is invalid.
        """
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})."
            )
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_p)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_p=dropout_p,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(p=dropout_p)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize tokens and linear projections following ViT best practices."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, return_last_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass mapping (B, C, H, W) cutout to class logits (B, K)."""
        batch_size, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected input spatial size ({self.image_size}, {self.image_size}), "
                f"got ({height}, {width})."
            )

        # Patch tokenization: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)

        last_attn: torch.Tensor | None = None
        for idx, block in enumerate(self.blocks):
            is_last = idx == len(self.blocks) - 1
            if return_last_attention and is_last:
                tokens, last_attn = block(tokens, return_attention=True)
            else:
                tokens = block(tokens, return_attention=False)

        cls_out = self.norm(tokens[:, 0])
        logits = self.head(self.head_drop(cls_out))

        if return_last_attention and last_attn is not None:
            return logits, last_attn
        return logits
