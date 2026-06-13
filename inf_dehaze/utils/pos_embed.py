# Position embedding utilities (adapted from Meta MAE / Llama RMSNorm).

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn


@lru_cache(maxsize=32)
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, i=0, j=0):
    """Return a cached 2-D sinusoidal position embedding."""
    return get_2d_sincos_pos_embed_base(embed_dim, grid_size, cls_token, i, j)


def get_2d_sincos_pos_embed_base(embed_dim, grid_size, cls_token=False, i=0, j=0):
    grid_h = np.arange(grid_size, dtype=np.float32) + i * grid_size
    grid_w = np.arange(grid_size, dtype=np.float32) + j * grid_size
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class LlamaRMSNorm(nn.Module):
    """Root-mean-square layer normalization (Llama-style)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
