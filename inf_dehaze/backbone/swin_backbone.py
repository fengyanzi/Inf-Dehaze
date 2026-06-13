# ---------------------------------------------------------------------------
# model/backbone/swin_backbone.py
#
# Swin-Transformer backbone for InfDehaze.
# Provides:
#   - SwinTransformerBlock  (unchanged W-MSA / SW-MSA block)
#   - PatchMerging          (2 modes: constant-dim | doubling)
#   - PatchExpandConv       (2 modes: constant-dim | halving)
#   - LinearEmbed           (patch stem)
#   - LinearUnembed         (final ConvTranspose2d upsample to pixel space)
#   - BasicEncoderStage     (one encoder stage: blocks + optional downsample)
#   - BasicDecoderStage     (one decoder stage: skip-concat + blocks + optional upsample)
#   - InfEncoder
#   - InfDecoder
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_

__all__ = [
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchExpandConv",
    "LinearEmbed",
    "LinearUnembed",
    "BasicEncoderStage",
    "BasicDecoderStage",
    "InfEncoder",
    "InfDecoder",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) → (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(num_windows*B, window_size, window_size, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Core attention & MLP
# ─────────────────────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block (W-MSA or SW-MSA)."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # Attention mask for SW-MSA
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, window_size={self.window_size}, "
            f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Patch down/up-sampling
# ─────────────────────────────────────────────────────────────────────────────

class PatchMerging(nn.Module):
    """Spatial downsampling (2×) with optional channel doubling.

    Args:
        dim:        Input channel count.
        double_dim: If True, output channels = 2*dim (standard Swin).
                    If False, output channels = dim (constant-dim mode).
        norm_layer: Normalization applied before the projection.
    """

    def __init__(self, dim: int, double_dim: bool = True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.double_dim = double_dim
        out_dim = dim * 2 if double_dim else dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    @property
    def out_dim(self) -> int:
        return self.dim * 2 if self.double_dim else self.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  →  (B, out_dim, H/2, W/2)"""
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0
        x = x.permute(0, 2, 3, 1)  # B H W C
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B H/2 W/2 4C
        x = self.norm(x)
        x = self.reduction(x)                     # B H/2 W/2 out_dim
        x = x.permute(0, 3, 1, 2).contiguous()   # B out_dim H/2 W/2
        return x


class PatchExpandConv(nn.Module):
    """Spatial upsampling (2×) with optional channel halving.

    Args:
        dim:       Input channel count.
        half_dim:  If True, output channels = dim // 2 (mirrors doubling).
                   If False, output channels = dim (constant-dim mode).
    """

    def __init__(self, dim: int, half_dim: bool = True):
        super().__init__()
        self.dim = dim
        self.half_dim = half_dim
        out_dim = dim // 2 if half_dim else dim
        self.expand = nn.ConvTranspose2d(dim, out_dim, kernel_size=2, stride=2)

    @property
    def out_dim(self) -> int:
        return self.dim // 2 if self.half_dim else self.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) token sequence  →  (B, L*4, C_out) token sequence.
        Internally converts to spatial, upsamples, then back to sequence.
        The caller supplies the spatial dims via the input_resolution attribute
        set on the parent stage.
        """
        raise RuntimeError("Call forward_spatial instead for spatial tensors.")

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  →  (B, out_dim, H*2, W*2)"""
        return self.expand(x)


class LinearEmbed(nn.Module):
    """RGB image → patch tokens via a strided convolution stem."""

    def __init__(self, patch_size: int = 2, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, embed_dim, H/ps, W/ps)"""
        return self.proj(x)


class LinearUnembed(nn.Module):
    """Final stage: tokens → pixel-space RGB via ConvTranspose2d."""

    def __init__(self, dim: int, output_dim: int = 3, patch_size: int = 2):
        super().__init__()
        self.expand = nn.ConvTranspose2d(dim, output_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) sequence  →  (B, output_dim, H*ps, W*ps) image."""
        # We receive a sequence; caller must ensure spatial shape is stored.
        raise RuntimeError("Use forward_spatial for spatial tensors.")

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  →  (B, output_dim, H*ps, W*ps)"""
        return self.expand(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder stage
# ─────────────────────────────────────────────────────────────────────────────

class BasicEncoderStage(nn.Module):
    """One encoder stage: Swin blocks → optional PatchMerging.

    The stage receives feature maps in (B, C, H, W) spatial format and
    returns (B, C_out, H_out, W_out).  The skip-connection output is
    produced *before* downsampling.

    Args:
        dim:            Input channel count for this stage.
        input_resolution: (H, W) after the patch stem.
        depth:          Number of SwinTransformerBlocks.
        num_heads:      Attention heads.
        window_size:    Local window size.
        mlp_ratio:      MLP expansion factor.
        drop_path:      Per-block drop-path rates (list) or scalar.
        double_dim:     Passed to PatchMerging; True → output 2×channels.
        downsample:     If True, append a PatchMerging at the end.
        use_checkpoint: Gradient checkpointing.
        norm_layer:     Normalization class.
        block_type:     "swin" (only built-in). Extend here for "conv" etc.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: Union[float, List[float]] = 0.0,
        double_dim: bool = True,
        downsample: bool = True,
        use_checkpoint: bool = False,
        norm_layer=nn.LayerNorm,
        block_type: str = "swin",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        # ── blocks ──────────────────────────────────────────────────────────
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dp = drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path
            if block_type == "swin":
                self.blocks.append(
                    SwinTransformerBlock(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if i % 2 == 0 else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        drop_path=dp,
                        norm_layer=norm_layer,
                    )
                )
            else:
                raise ValueError(f"Unknown block_type '{block_type}'. Add it here.")

        # ── optional downsample ──────────────────────────────────────────────
        self.patch_merging: Optional[PatchMerging] = None
        if downsample:
            self.patch_merging = PatchMerging(dim=dim, double_dim=double_dim, norm_layer=norm_layer)

    @property
    def out_dim(self) -> int:
        if self.patch_merging is not None:
            return self.patch_merging.out_dim
        return self.dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            skip: (B, C, H, W)   — pre-downsample feature map (skip connection)
            out:  (B, C_out, H/2, W/2) or (B, C, H, W) if no downsample
        """
        B, C, H, W = x.shape
        # Flatten spatial → sequence for Swin blocks
        x_seq = x.flatten(2).transpose(1, 2)  # B, H*W, C

        for blk in self.blocks:
            if self.use_checkpoint:
                x_seq = cp.checkpoint(blk, x_seq)
            else:
                x_seq = blk(x_seq)

        # Back to spatial
        x = x_seq.transpose(1, 2).view(B, C, H, W)
        skip = x  # skip BEFORE downsampling

        if self.patch_merging is not None:
            x = self.patch_merging(x)

        return skip, x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}"


# ─────────────────────────────────────────────────────────────────────────────
# Decoder stage
# ─────────────────────────────────────────────────────────────────────────────

class BasicDecoderStage(nn.Module):
    """One decoder stage: skip-concat → optional PatchExpandConv → Swin blocks.

    The stage receives:
      - ``x``: current feature map (B, C, H, W)
      - ``skip``: encoder skip (B, C_skip, H, W) — same spatial size as x

    The skip is concatenated along channels, projected down to ``out_dim``,
    then processed by Swin blocks.  Finally an optional upsample doubles H/W.

    Args:
        in_dim:         Channels of the incoming ``x``.
        skip_dim:       Channels of the skip connection.
        out_dim:        Channels after projection / throughout blocks.
        input_resolution: Spatial size (H, W) of the *input* to this stage.
        depth, num_heads, window_size, mlp_ratio, drop_path, norm_layer:
            Standard Swin parameters.
        upsample:       If True, append a PatchExpandConv at the end.
        half_dim:       Passed to PatchExpandConv (True → halve channels).
        use_checkpoint: Gradient checkpointing in blocks.
        block_type:     Extension hook — currently only "swin".
    """

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        out_dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: Union[float, List[float]] = 0.0,
        upsample: bool = True,
        half_dim: bool = True,
        use_checkpoint: bool = False,
        norm_layer=nn.LayerNorm,
        block_type: str = "swin",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.skip_dim = skip_dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        # Project skip-concatenated channels down to out_dim
        self.proj = nn.Linear(in_dim + skip_dim, out_dim, bias=False)

        # Swin blocks at out_dim
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dp = drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path
            if block_type == "swin":
                self.blocks.append(
                    SwinTransformerBlock(
                        dim=out_dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if i % 2 == 0 else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        drop_path=dp,
                        norm_layer=norm_layer,
                    )
                )
            else:
                raise ValueError(f"Unknown block_type '{block_type}'. Add it here.")

        # Optional upsample
        self.patch_expand: Optional[PatchExpandConv] = None
        if upsample:
            self.patch_expand = PatchExpandConv(dim=out_dim, half_dim=half_dim)

    @property
    def final_out_dim(self) -> int:
        if self.patch_expand is not None:
            return self.patch_expand.out_dim
        return self.out_dim

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, in_dim,   H, W)
            skip: (B, skip_dim, H, W)
        Returns:
            out:  (B, final_out_dim, H*2, W*2) if upsample else (B, out_dim, H, W)
        """
        B, _, H, W = x.shape
        # Concat & project in token-sequence space
        x_seq = x.flatten(2).transpose(1, 2)        # B, H*W, in_dim
        s_seq = skip.flatten(2).transpose(1, 2)     # B, H*W, skip_dim
        x_seq = self.proj(torch.cat([x_seq, s_seq], dim=-1))  # B, H*W, out_dim

        for blk in self.blocks:
            if self.use_checkpoint:
                x_seq = cp.checkpoint(blk, x_seq)
            else:
                x_seq = blk(x_seq)

        # Back to spatial
        x = x_seq.transpose(1, 2).view(B, self.out_dim, H, W)

        if self.patch_expand is not None:
            x = self.patch_expand.forward_spatial(x)

        return x

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, skip_dim={self.skip_dim}, out_dim={self.out_dim}, "
            f"input_resolution={self.input_resolution}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full Encoder
# ─────────────────────────────────────────────────────────────────────────────

class InfEncoder(nn.Module):
    """Hierarchical Swin encoder.

    Dim schedule:
      Stage 0: base_dim           (PatchMerging with double_dim=False — no doubling)
      Stage 1: base_dim * 2       (double_dim=True)
      Stage 2: base_dim * 4
      …

    The last stage has no downsample (it feeds directly into the bottleneck).

    Args:
        base_dim:    Base embedding dimension.
        depths:      Depth per stage.
        num_heads:   Attention heads per stage.
        img_size:    Spatial size of crops fed to the encoder.
        window_size: Swin local window size.
        patch_size:  Stem conv stride (pixels).
        mlp_ratio:   MLP expansion factor.
        drop_path_rate: Max stochastic depth rate.
        use_checkpoint: Gradient checkpointing.
        block_types: List of block type strings per stage.
    """

    def __init__(
        self,
        base_dim: int = 96,
        depths: List[int] = [2, 3, 4, 2],
        num_heads: List[int] = [2, 4, 8, 8],
        img_size: int = 256,
        window_size: int = 8,
        patch_size: int = 2,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        block_types: Optional[List[str]] = None,
    ):
        super().__init__()
        num_stages = len(depths)
        assert len(num_heads) == num_stages

        self.num_stages = num_stages
        self.patch_size = patch_size
        self.img_size = img_size

        # Patch stem
        self.patch_embed = LinearEmbed(patch_size=patch_size, in_chans=3, embed_dim=base_dim)

        # Stage 0 and 1 share base_dim; deeper stages double channels.
        self.dim_schedule = [base_dim]
        for i in range(1, num_stages):
            if i == 1:
                self.dim_schedule.append(self.dim_schedule[-1])
            else:
                self.dim_schedule.append(self.dim_schedule[-1] * 2)

        # Drop-path schedule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        if block_types is None:
            block_types = ["swin"] * num_stages

        current_res = img_size // patch_size
        self.stages = nn.ModuleList()
        depth_offset = 0

        for i in range(num_stages):
            dim = self.dim_schedule[i]
            depth = depths[i]
            is_last = i == num_stages - 1

            # Stage 0: no channel doubling; all later stages: double
            double_dim = i > 0

            stage = BasicEncoderStage(
                dim=dim,
                input_resolution=(current_res, current_res),
                depth=depth,
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[depth_offset: depth_offset + depth],
                double_dim=double_dim,
                downsample=not is_last,
                use_checkpoint=use_checkpoint,
                block_type=block_types[i],
            )
            self.stages.append(stage)
            depth_offset += depth
            if not is_last:
                current_res //= 2

        self.bottleneck_dim = self.dim_schedule[-1]  # convenient accessor

    # -----------------------------------------------------------------
    def _run_stages(self, x: torch.Tensor):
        """Run all encoder stages; return (skip_list, last_feature)."""
        skips = []
        for stage in self.stages:
            skip, x = stage(x)
            skips.append(skip)
        return skips, x

    def forward(
        self, x: torch.Tensor, train: bool = False
    ):
        """
        Args:
            x:     (B, 3, H, W)
            train: If True, skip tensors stay on GPU.
                   If False, skip tensors are moved to CPU to free VRAM.

        Returns:
            skips: list of skip tensors, one per stage
            last:  (B, C_last, H_last, W_last) on GPU
        """
        x = self.patch_embed(x)  # (B, base_dim, H/ps, W/ps)

        if train:
            skips, last = self._run_stages(x)
        else:
            # Inference: offload skips to CPU to save VRAM
            raw_skips, last = self._run_stages(x)
            skips = [s.cpu() for s in raw_skips]

        return skips, last

    def extra_repr(self) -> str:
        return f"dim_schedule={self.dim_schedule}"


# ─────────────────────────────────────────────────────────────────────────────
# Full Decoder
# ─────────────────────────────────────────────────────────────────────────────

class InfDecoder(nn.Module):
    """Hierarchical Swin decoder that mirrors InfEncoder.

    Dim schedule (mirrors encoder in reverse):
      Stage 0 of decoder operates at encoder's last dim (no halving at end).
      Intermediate stages halve channels after blocks.
      Last decoder stage keeps constant channels (mirrors encoder stage 0).

    Args:
        encoder_dim_schedule: Ordered list of dims from InfEncoder (lo→hi).
        depths:    Depth per decoder stage (typically reversed encoder depths).
        num_heads: Attention heads per stage.
        window_size: Swin window size.
        mlp_ratio, drop_path_rate, use_checkpoint, block_types: standard params.
        patch_size: Stem stride; used by LinearUnembed at the end.
        out_channels: Final output channels (e.g. 3 for RGB).
    """

    def __init__(
        self,
        encoder_dim_schedule: List[int],
        depths: List[int],
        num_heads: List[int],
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        block_types: Optional[List[str]] = None,
        patch_size: int = 2,
        out_channels: int = 3,
        crop_size: int = 256,
    ):
        super().__init__()
        num_stages = len(depths)
        assert len(num_heads) == num_stages
        assert len(encoder_dim_schedule) == num_stages

        # encoder_dim_schedule goes from shallowest (base_dim) to deepest
        # decoder mirrors it: deepest → shallowest
        dec_dims = list(reversed(encoder_dim_schedule))   # deep → shallow
        self.dim_schedule = encoder_dim_schedule[::-1]
        self.num_stages = num_stages

        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        if block_types is None:
            block_types = ["swin"] * num_stages

        # Spatial resolutions mirror encoder (encoder produces half per stage)
        # dec stage i receives spatial = crop_size/patch_size / 2^(num_stages-1-i)
        base_res = crop_size // patch_size
        enc_resolutions = [base_res // (2 ** i) for i in range(num_stages)]
        # decoder: stage 0 at enc_resolutions[-1], …, stage n-1 at enc_resolutions[0]
        dec_resolutions = list(reversed(enc_resolutions))

        self.stages = nn.ModuleList()
        depth_offset = 0

        for i in range(num_stages):
            dim = self.dim_schedule[i]
            in_dim = dec_dims[i]       # incoming feature dim
            skip_dim = dec_dims[i]     # skip dim = same encoder level
            # Output dim after projection: same as in_dim (halving happens in upsample)
            out_dim = dec_dims[i]
            is_last = i == num_stages - 1

            if i < num_stages - 1:
                half_dim = self.dim_schedule[i + 1] < dim
            else:
                half_dim = False

            stage = BasicDecoderStage(
                in_dim=in_dim,
                skip_dim=skip_dim,
                out_dim=out_dim,
                input_resolution=(dec_resolutions[i], dec_resolutions[i]),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[depth_offset: depth_offset + depths[i]],
                upsample=not is_last,
                half_dim=half_dim,
                use_checkpoint=use_checkpoint,
                block_type=block_types[i],
            )
            self.stages.append(stage)
            depth_offset += depths[i]

        # Final pixel-space upsample
        self.unembed = LinearUnembed(
            dim=dec_dims[-1],
            output_dim=out_channels,
            patch_size=patch_size,
        )

    # -----------------------------------------------------------------
    def _decode_batch(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
    ) -> torch.Tensor:
        """Run decoder stages on a single spatial feature tensor."""
        # Skips are in encoder order (shallow → deep).
        # Decoder consumes them deep → shallow.
        for i, stage in enumerate(self.stages):
            skip_idx = self.num_stages - 1 - i  # deep → shallow
            skip = skips[skip_idx]
            if skip.device != x.device:
                skip = skip.to(x.device)
            x = stage(x, skip)
        x = self.unembed.forward_spatial(x)
        return x

    # -----------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        n_regions: int,
        batch_size: int,
        train: bool = False,
    ) -> torch.Tensor:
        if train:
            return self._training_forward(x, skips, n_regions, batch_size)
        else:
            return self._inference_forward(x, skips, n_regions, batch_size)

    # -----------------------------------------------------------------
    def _training_forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        n_regions: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        x:    (B, C, HP*HC, WP*WC)  bottleneck output (reassembled)
        skips: list[tensor] — on GPU, shape (N_patches, C_i, H_i, W_i)
        """
        # Reassemble and split into patch-batch dimension
        x = rearrange(x, "N C (HP HC) (WP WC) -> (N HP WP) C HC WC",
                      HP=n_regions, WP=n_regions)
        total = x.shape[0]
        outputs = []

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            x_b = x[start:end]
            # Slice each skip to the same patch indices
            skips_b = [s[start:end] for s in skips]
            out = self._decode_batch(x_b, skips_b)
            outputs.append(out)

        out = torch.cat(outputs, dim=0)
        out = rearrange(out, "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
                        HP=n_regions, WP=n_regions)
        return out

    # -----------------------------------------------------------------
    def _inference_forward(
        self,
        x: torch.Tensor,
        skips: List[List[torch.Tensor]],
        n_regions: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        x:    (B, C, HP*HC, WP*WC)  bottleneck output
        skips: list of lists — skips[level][batch_idx] on CPU
        """
        x = rearrange(x, "N C (HP HC) (WP WC) -> (N HP WP) C HC WC",
                      HP=n_regions, WP=n_regions)
        total = x.shape[0]
        outputs = []

        for batch_idx, start in enumerate(range(0, total, batch_size)):
            end = min(start + batch_size, total)
            x_b = x[start:end]
            # Retrieve the correct batch chunk for each level
            skips_b = [skips[level][batch_idx] for level in range(self.num_stages)]
            out = self._decode_batch(x_b, skips_b)
            outputs.append(out.cpu())

        del x
        outputs = [o.to(next(self.parameters()).device) for o in outputs]
        out = torch.cat(outputs, dim=0)
        out = rearrange(out, "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
                        HP=n_regions, WP=n_regions)
        return out
