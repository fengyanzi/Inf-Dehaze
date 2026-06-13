"""Local Modeling Mixture-of-Experts (LMMoE).

Spatial convolution experts with top-k routing and a shared expert branch.
Used as the local modeling complement in every bottleneck block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseConv(nn.Module):
    """Depthwise separable convolution with GELU activation."""

    def __init__(self, hidden_features, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=hidden_features,
            bias=False,
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class ConvExpert(nn.Module):
    """Single convolution expert: 1x1 -> spatial conv -> 1x1."""

    def __init__(self, in_channels, out_channels, hidden_channels, conv_type):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.act = nn.GELU()

        conv_factories = {
            "3x3": lambda: nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            "5x5": lambda: nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2, bias=False),
            "7x7": lambda: nn.Conv2d(hidden_channels, hidden_channels, 7, padding=3, bias=False),
            "dw3x3": lambda: DepthWiseConv(hidden_channels, kernel_size=3),
            "dw5x5": lambda: DepthWiseConv(hidden_channels, kernel_size=5),
            "dw7x7": lambda: DepthWiseConv(hidden_channels, kernel_size=7),
        }
        if conv_type not in conv_factories:
            raise ValueError(f"Unknown conv type: {conv_type}")
        self.conv2 = conv_factories[conv_type]()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.conv3.weight, a=math.sqrt(5))
        if isinstance(self.conv2, nn.Conv2d):
            nn.init.kaiming_uniform_(self.conv2.weight, a=math.sqrt(5))
        elif isinstance(self.conv2, DepthWiseConv):
            nn.init.kaiming_uniform_(self.conv2.conv.weight, a=math.sqrt(5))

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.act(x)
        return self.conv3(x)


class MoEGate2D(nn.Module):
    """Top-k gating over spatial feature maps."""

    def __init__(self, in_channels, n_experts, k=2, aux_loss_alpha=0.01):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.aux_loss_alpha = aux_loss_alpha
        self.gate = nn.Conv2d(in_channels, n_experts, kernel_size=1)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))

    def forward(self, x, train=False):
        scores = self.gate(x).permute(0, 2, 3, 1)
        scores = F.softmax(scores, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, k=self.k, dim=-1)

        if self.k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        if train and self.aux_loss_alpha > 0:
            expert_counts = torch.bincount(topk_indices.view(-1), minlength=self.n_experts).float()
            expert_probs = scores.mean(dim=(0, 1, 2))
            aux_loss = (expert_probs * expert_counts).sum() * self.aux_loss_alpha
        else:
            aux_loss = None

        return topk_indices, topk_weights, aux_loss


class AddAuxiliaryLoss(nn.Module):
    """Attach auxiliary MoE loss to the forward graph without changing values."""

    def forward(self, x, loss):
        return x if loss is None else x + 0 * loss


class LMMoE(nn.Module):
    """Local Modeling Mixture-of-Experts (LMMoE)."""

    def __init__(
        self,
        in_channels=128,
        out_channels=128,
        hidden_channels=512,
        n_experts=12,
        k=3,
        aux_loss_alpha=0.01,
        n_shared_experts=1,
        shared_hidden_channels=None,
    ):
        super().__init__()
        assert n_experts % 3 == 0, "n_experts must be divisible by 3"
        self.out_channels = out_channels
        self.k = k

        self.experts = nn.ModuleList()
        experts_per_type = n_experts // 3
        for conv_type in ("dw3x3", "dw5x5", "dw7x7"):
            for _ in range(experts_per_type):
                self.experts.append(
                    ConvExpert(in_channels, out_channels, hidden_channels, conv_type)
                )

        if n_shared_experts > 0:
            shared_hidden = shared_hidden_channels or hidden_channels * n_shared_experts
            self.shared_experts = ConvExpert(
                in_channels, out_channels, shared_hidden, "5x5"
            )
        else:
            self.shared_experts = None

        self.gate = MoEGate2D(in_channels, n_experts, k, aux_loss_alpha)
        self.add_loss = AddAuxiliaryLoss()

    def forward(self, x, train=False):
        batch, _, width, height = x.shape
        expert_indices, expert_weights, aux_loss = self.gate(x, train)
        output = torch.zeros(batch, self.out_channels, width, height, device=x.device, dtype=x.dtype)

        for expert_idx, expert in enumerate(self.experts):
            mask = (expert_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue

            expert_out = expert(x)
            weights = torch.zeros(batch, width, height, device=x.device, dtype=x.dtype)
            for route in range(self.k):
                weights += torch.where(
                    expert_indices[..., route] == expert_idx,
                    expert_weights[..., route],
                    torch.zeros_like(expert_weights[..., route]),
                )
            output += expert_out * weights.unsqueeze(1)

        if self.shared_experts is not None:
            output = output + self.shared_experts(x)

        output = self.add_loss(output, aux_loss)
        if train:
            return output, aux_loss
        return output
