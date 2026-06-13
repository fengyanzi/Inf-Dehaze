"""Intra-Block Sparse Self-Attention (IBSSA).

Efficient approximate global attention for the bottleneck via LSH-based
block sorting plus a low-rank residual sampled from uniform key indices.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IBSSA(nn.Module):
    """Intra-Block Sparse Self-Attention (IBSSA)."""

    def __init__(
        self,
        dim=384,
        lsh_num_projs=7,
        num_heads=4,
        block_size=128,
        sample_size=128,
        min_seq_len=8192,
        qk_dim=128,
    ):
        super().__init__()
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        assert dim % num_heads == 0, (dim, num_heads)
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.dim = dim
        self.lsh = AngularLSH(
            num_projs=self.lsh_num_projs,
            dim=(1, 1, qk_dim // num_heads),
        )

    def forward(self, x, scale=None):
        b, length, _ = x.shape
        query = self.to_q(x).reshape(b, length, self.num_heads, -1).permute(0, 2, 1, 3)
        key = self.to_k(x).reshape(b, length, self.num_heads, -1).permute(0, 2, 1, 3)
        value = self.to_v(x).reshape(b, length, self.num_heads, -1).permute(0, 2, 1, 3)
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        head_dim = key.shape[-1]
        scale = head_dim ** (-0.5) if scale is None else scale

        attn = self._forward_hda(query, key, value, scale).permute(0, 2, 1, 3)
        return attn.reshape(b, length, -1).contiguous()

    def _forward_hda(self, query, key, value, scale):
        batch_size, _, n_query, _ = query.shape
        n_key = key.shape[2]

        if self.min_seq_len > n_query:
            return _exact_attention(query, key, value, scale)

        _, query_sort_idx = torch.sort(self.lsh.hash(query), dim=2, stable=True)
        _, key_sort_idx = torch.sort(self.lsh.hash(key), dim=2, stable=True)
        query_sort_idx_inv = torch.argsort(query_sort_idx, dim=2, stable=True)

        key_block_size = self.block_size
        query_sorted = _indexing(query, query_sort_idx, key_block_size)
        key_sorted = _indexing(key, key_sort_idx, key_block_size)
        value_sorted = _indexing(value, key_sort_idx, key_block_size)

        if key_block_size > 0:
            num_blocks = key_sorted.shape[2] // key_block_size
            query_block_size = query_sorted.shape[2] // num_blocks

            query_split = query_sorted.view(-1, 1, query_block_size, self.qk_dim // self.num_heads)
            key_split = key_sorted.view(-1, 1, key_block_size, self.qk_dim // self.num_heads)
            value_split = value_sorted.view(-1, 1, key_block_size, self.dim // self.num_heads)

            attn_block = _exact_attention(query_split, key_split, value_split, scale)
            if attn_block.shape[2] not in attn_block.stride():
                attn_block = attn_block.contiguous()
            attn_block = attn_block.view(batch_size, self.num_heads, query_sorted.shape[2], -1)

            if query_sorted.shape[2] != n_query:
                attn_block = attn_block[:, :, :n_query, :]
        else:
            query_block_size = -1
            attn_block = 0

        sample_size = self.sample_size
        if sample_size > 0 and (n_query > query_block_size) and (n_key > key_block_size):
            sampled_set = torch.randint(
                n_key,
                size=(batch_size, self.num_heads, sample_size),
                device=query_sorted.device,
            )
            value_subset = _indexing(value_sorted, sampled_set)
            key_subset = _indexing(key_sorted, sampled_set)
            attn_res = _exact_attention(query_sorted, key_subset, value_subset, scale)
            attn = attn_block + attn_res.mul(0.1) if key_block_size > 0 else attn_res
        else:
            attn = attn_block

        return _indexing(attn, query_sort_idx_inv)


class AngularLSH(nn.Module):
    """Angular locality-sensitive hashing for token sorting."""

    def __init__(self, num_projs, dim, rng=None):
        super().__init__()
        self.num_projs = num_projs
        if num_projs > 0:
            self.register_buffer(
                "proj_dir",
                torch.randn(dim + (num_projs,), generator=rng),
                persistent=False,
            )
            self.register_buffer(
                "perm",
                self._unit_hamming_distance_array(self.num_projs),
                persistent=False,
            )
            self.register_buffer(
                "enc_vec",
                2 ** torch.arange(self.num_projs).view(1, 1, 1, -1),
                persistent=False,
            )

    def _unit_hamming_distance_array(self, size_n):
        if size_n == 1:
            return torch.tensor([0, 1])
        arr = self._unit_hamming_distance_array(size_n - 1)
        return torch.concat([arr, torch.flip(arr, dims=[0]) + 2 ** (size_n - 1)], 0)

    def hash(self, mat):
        if self.num_projs < 0:
            return torch.zeros(mat.shape[:-1], device=mat.device, dtype=torch.int32)
        mask = torch.einsum("...nd,...dr -> ...nr", mat, self.proj_dir) > 0
        bin_ids = (mask * self.enc_vec).sum(-1)
        return self.perm[bin_ids]


def _indexing(x, indices, chunk_size=-1):
    """Gather tokens along the sequence dimension."""
    if chunk_size < 0 or (chunk_size > 0 and x.shape[-2] % chunk_size == 0):
        return x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))

    x = x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
    new_n = math.ceil(x.shape[2] / chunk_size) * chunk_size
    if new_n <= 0 or new_n - x.shape[2] <= 0:
        raise ValueError(
            f"Invalid padding size when chunk_size={chunk_size}, seq_len={x.shape[2]}"
        )
    return F.pad(x, (0, 0, 0, new_n - x.shape[2]), mode="constant", value=0.0)


def _exact_attention(q, k, v, softmax_scale):
    del softmax_scale  # scaled_dot_product_attention applies scaling internally
    return F.scaled_dot_product_attention(q, k, v)
