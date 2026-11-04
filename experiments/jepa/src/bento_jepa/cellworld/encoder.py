"""Spatial transformer used by the context, target, and predictor arms."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_encoding import ALiBi2D


class SpatialSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        spatial_cfg: Optional[dict] = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)
        self.spatial_cfg = spatial_cfg or {}

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.position_module = ALiBi2D(
            num_heads=num_heads,
            scale=self.spatial_cfg.get("alibi_scale", 1.0),
        )

    def _split_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tokens.shape
        return tokens.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _merge_heads(self, token_heads: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_tokens, _ = token_heads.shape
        return token_heads.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.dim
        )

    def _padding_bias(
        self,
        key_padding_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if key_padding_mask is None:
            return None
        padding_bias = torch.zeros_like(key_padding_mask, dtype=dtype)
        padding_bias.masked_fill_(key_padding_mask, float("-inf"))
        return padding_bias[:, None, None, :]

    def forward(
        self,
        cell_tokens: torch.Tensor,
        coordinates: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if coordinates is None:
            raise ValueError("SpatialSelfAttention requires coordinates")

        query_projection, key_projection, value_projection = self.qkv(
            cell_tokens
        ).chunk(3, dim=-1)
        query_heads = self._split_heads(query_projection)
        key_heads = self._split_heads(key_projection)
        value_heads = self._split_heads(value_projection)

        attention_bias = self.position_module(coordinates).to(
            device=query_heads.device, dtype=query_heads.dtype
        )

        padding_bias = self._padding_bias(key_padding_mask, query_heads.dtype)
        if padding_bias is not None:
            attention_bias = attention_bias + padding_bias

        attention_output = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        merged_output = self._merge_heads(attention_output)
        return self.proj(merged_output)


class SpatialTransformerBlock(nn.Module):
    """Pre-norm spatial cell-attention block with shape-preserving residuals.

    D is the cell embedding dimension and H = int(D * mlp_ratio) is the MLP
    hidden dimension. Both attention and the complete block map [B, N, D] back
    to [B, N, D].
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        spatial_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialSelfAttention(dim, num_heads, dropout, spatial_cfg)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)  # H
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # [..., D] -> [..., H]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),  # [..., H] -> [..., D]
            nn.Dropout(dropout),
        )

    def forward(
        self,
        cell_tokens: torch.Tensor,
        coordinates: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transform one spatial cell sequence without changing its shape.

        Args:
            cell_tokens: [B, N, D] tokens providing attention queries/K/V.
            coordinates: [B, N, 2] spatial positions used to build 2D-ALiBi.
            key_padding_mask: Optional [B, N] mask; True excludes that cell
                from attention keys and values.

        Returns:
            [B, N, D] updated cell tokens.
        """
        normalized_cell_tokens = self.norm1(cell_tokens)  # [B, N, D]
        attention_output = self.attn(
            normalized_cell_tokens,
            coordinates=coordinates,
            key_padding_mask=key_padding_mask,
        )  # [B, N, D]
        cell_tokens = cell_tokens + attention_output  # [B, N, D]

        mlp_output = self.mlp(self.norm2(cell_tokens))  # [B, N, D] via H
        return cell_tokens + mlp_output  # [B, N, D]


class SpatialTransformerStack(nn.Module):
    """A stack whose module names match the canonical bag checkpoints."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        spatial_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    spatial_cfg,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        cell_tokens: torch.Tensor,
        coordinates: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            cell_tokens = block(
                cell_tokens,
                coordinates=coordinates,
                key_padding_mask=key_padding_mask,
            )
        return self.norm(cell_tokens)
