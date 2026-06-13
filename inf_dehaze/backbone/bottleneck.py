"""Bottleneck with IBSSA / standard attention and LMMoE local branches."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from inf_dehaze.attention.ibssa import IBSSA
from inf_dehaze.attention.lmmoe import LMMoE
from inf_dehaze.utils.pos_embed import LlamaRMSNorm, get_2d_sincos_pos_embed

__all__ = ["INFBottleneck"]

_VALID_LAYER_TYPES = {"ibssa", "sa"}


class LlamaMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _IBSSACore(nn.Module):
    """IBSSA attention core."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = IBSSA(
            dim=dim,
            num_heads=num_heads,
            lsh_num_projs=7,
            block_size=128,
            sample_size=128,
            min_seq_len=8192,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


class _SACore(nn.Module):
    """Standard scaled dot-product attention core."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, channels = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, length, channels))


class BottleneckBlock(nn.Module):
    """One bottleneck block: global attention + global MLP + LMMoE local branch."""

    def __init__(
        self,
        dim: int,
        layer_type: str,
        num_heads: int,
        mlp_ratio: int = 4,
        moe_scale: float = 0.1,
    ):
        super().__init__()
        assert layer_type in _VALID_LAYER_TYPES
        self.layer_type = layer_type
        self.moe_scale = moe_scale

        self.ln1 = LlamaRMSNorm(dim, eps=1e-5)
        self.ln2 = LlamaRMSNorm(dim, eps=1e-5)
        self.attn_core = _IBSSACore(dim, num_heads) if layer_type == "ibssa" else _SACore(dim, num_heads)
        self.mlp = LlamaMLP(dim, dim * mlp_ratio)
        self.moe = LMMoE(in_channels=dim, out_channels=dim, hidden_channels=dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        h: int,
        w: int,
        train: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        x_seq = x_seq + self.attn_core(self.ln1(x_seq))

        x_2d = rearrange(x_seq, "b (h w) c -> b c h w", h=h, w=w)
        moe_loss = 0.0
        if train:
            local_2d, moe_loss = self.moe(x_2d, train=True)
        else:
            local_2d = self.moe(x_2d)
        local_seq = rearrange(local_2d, "b c h w -> b (h w) c") * self.moe_scale

        x_seq = x_seq + self.mlp(self.ln2(x_seq)) + local_seq
        return x_seq, moe_loss


class INFBottleneck(nn.Module):
    """Configurable bottleneck: reassemble tiles, add sincos PE, run stacked blocks."""

    def __init__(
        self,
        in_dim: int,
        n_layers: int = 4,
        layer_types: Optional[List[str]] = None,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        moe_scale: float = 0.1,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        if hidden_size is not None and hidden_size != in_dim:
            raise ValueError("hidden_size must equal in_dim")

        self.in_dim = in_dim
        self.hidden_size = in_dim

        if not layer_types:
            resolved = ["ibssa"] * n_layers
        else:
            if len(layer_types) != n_layers:
                raise ValueError("layer_types length must equal n_layers")
            resolved = [t.lower() for t in layer_types]
            invalid = [t for t in resolved if t not in _VALID_LAYER_TYPES]
            if invalid:
                raise ValueError(f"Unknown layer type(s): {invalid}")

        self.layer_types = resolved
        self.blocks = nn.ModuleList([
            BottleneckBlock(
                dim=in_dim,
                layer_type=resolved[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                moe_scale=moe_scale,
            )
            for i in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        n_regions: int,
        train: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        x = rearrange(
            x,
            "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
            HP=n_regions,
            WP=n_regions,
        )
        _, channels, height, width = x.shape

        pos_embed = get_2d_sincos_pos_embed(channels, height, cls_token=False)
        x_seq = rearrange(x, "b c h w -> b (h w) c")
        x_seq = x_seq + torch.tensor(pos_embed, dtype=x_seq.dtype, device=x_seq.device)

        total_moe_loss = 0.0
        for block in self.blocks:
            x_seq, moe_loss = block(x_seq, height, width, train=train)
            total_moe_loss += moe_loss

        x = rearrange(x_seq, "b (h w) c -> b c h w", h=height, w=width)
        if train:
            return x, total_moe_loss
        return x
