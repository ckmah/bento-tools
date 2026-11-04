import math

import torch
import torch.nn as nn


def normalize_positions(
    positions: torch.Tensor,
    mode: str = "center_scale",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize [B, N, 2] coordinates for relative spatial attention."""
    if mode in ("none", None):
        return positions
    centered = positions - positions.mean(dim=1, keepdim=True)
    if mode == "center":
        return centered
    if mode == "center_std":
        scale = centered.flatten(1).std(dim=1).view(-1, 1, 1).clamp_min(eps)
        return centered / scale
    if mode == "center_scale":
        scale = centered.abs().amax(dim=(1, 2), keepdim=True).clamp_min(eps)
        return centered / scale
    if mode == "unit_box":
        lower_bound = positions.amin(dim=1, keepdim=True)
        upper_bound = positions.amax(dim=1, keepdim=True)
        max_extent = (upper_bound - lower_bound).amax(
            dim=-1, keepdim=True
        )
        return (positions - lower_bound) / max_extent.clamp_min(eps)
    raise ValueError(f"unknown position normalization mode: {mode}")


def pairwise_distance(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    delta = positions.unsqueeze(2) - positions.unsqueeze(1)   # [B, N, N, 2]
    return (delta.square().sum(dim=-1) + eps).sqrt()


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    def slopes_power_of_2(head_count: int):
        start = 2.0 ** (-(2.0 ** -(math.log2(head_count) - 3)))
        ratio = start
        return [
            start * ratio ** head_index
            for head_index in range(head_count)
        ]

    if math.log2(num_heads).is_integer():
        slopes = slopes_power_of_2(num_heads)
    else:
        closest_power_of_two = 2 ** math.floor(math.log2(num_heads))
        slopes = slopes_power_of_2(closest_power_of_two)
        slopes += slopes_power_of_2(2 * closest_power_of_two)[0::2][
            : num_heads - closest_power_of_two
        ]
    return torch.tensor(slopes, dtype=torch.float32)


class ALiBi2D(nn.Module):
    """2D ALiBi bias: per-head negative slope times Euclidean distance."""

    def __init__(self, num_heads: int, scale: float = 1.0):
        super().__init__()
        self.scale = float(scale)
        self.register_buffer("slopes", _alibi_slopes(num_heads), persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        distances = pairwise_distance(positions)
        head_slopes = -self.scale * self.slopes.to(
            distances.device, distances.dtype
        ).view(1, -1, 1, 1)
        return head_slopes * distances.unsqueeze(1)
