"""
network.py
===========
The neural network: a deep residual "trunk" (in the style of
AlphaZero / LeelaChessZero) that splits into three heads:

    Policy head      -> 4672-way move distribution (logits)
    Value head        -> win/draw/loss distribution (or scalar [-1,1])
    Aggression head   -> auxiliary scalar in [-1, 1] trained to predict
                          the hand-crafted "attacking style" heuristic
                          from heuristics.py. This is the concrete
                          implementation of the requested "custom
                          evaluation bias": rather than hard-coding a
                          fudge factor into the value, we give the
                          network a second task that *forces its
                          internal features* to represent material
                          activity / king safety / sacrifice potential,
                          and we additionally blend its output into the
                          value MCTS actually searches with (see
                          mcts.py: `effective_value`).

Squeeze-and-excitation (SE) blocks are included (toggle in config) —
modern Leela-style nets use them to let the network learn global,
whole-board context (useful for king-safety-style features) cheaply.
"""

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetworkConfig


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel-attention block."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels * 2)  # outputs gate (W) and bias (B)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        z = self.pool(x).view(b, c)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        gate, bias = torch.split(z, self.channels, dim=1)
        gate = torch.sigmoid(gate).view(b, c, 1, 1)
        bias = bias.view(b, c, 1, 1)
        return x * gate + bias


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        out = out + residual
        return F.relu(out)


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, policy_size: int):
        super().__init__()
        # "Full conv" style policy head: a 1x1 conv down to 73 planes
        # directly mirrors the 73-move-plane policy layout from
        # move_encoding.py, then we flatten to the flat 4672 vector.
        self.conv = nn.Conv2d(in_channels, 73, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)              # (B, 73, 8, 8)
        b = out.shape[0]
        # Flatten as (from_square * 73 + plane) to match move_encoding's index scheme:
        # move_encoding uses from_sq*73+plane, i.e. square-major, plane-minor.
        out = out.permute(0, 2, 3, 1).reshape(b, -1)  # (B, 8*8*73) with square-major order
        return out


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int, use_wdl: bool):
        super().__init__()
        self.use_wdl = use_wdl
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, hidden)
        out_dim = 3 if use_wdl else 1
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = out.flatten(1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        if self.use_wdl:
            return out  # raw logits over (win, draw, loss); softmax applied by caller
        return torch.tanh(out)  # scalar in [-1, 1]


class AggressionHead(nn.Module):
    """
    Small auxiliary head trained to regress the hand-crafted
    `heuristics.aggression_score` (see heuristics.py) directly from
    the trunk's features. Forces the trunk to encode attacking-style
    signals (king exposure, active pieces, sac potential) as first-
    class features rather than leaving them implicit, and gives MCTS
    a learned (not just hard-coded) source of attacking bias once
    training has converged.
    """

    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 8 * 8, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = out.flatten(1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        return out.squeeze(-1)  # (B,) in [-1, 1]


class ChessNet(nn.Module):
    """
    Full network: stem conv -> N residual (+SE) blocks -> three heads.

    forward() returns a dict with keys:
        "policy_logits"      : (B, 4672)
        "value"               : (B, 3) WDL logits, or (B,) scalar if use_wdl=False
        "aggression"          : (B,)   in [-1, 1]   (only if cfg.use_aggression_head)
    """

    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg

        self.stem_conv = nn.Conv2d(cfg.input_channels, cfg.num_filters, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(cfg.num_filters)

        self.tower = nn.ModuleList([
            ResidualBlock(cfg.num_filters, use_se=cfg.se_blocks)
            for _ in range(cfg.num_residual_blocks)
        ])

        self.policy_head = PolicyHead(cfg.num_filters, cfg.policy_size)
        self.value_head = ValueHead(cfg.num_filters, cfg.value_hidden_size, cfg.use_wdl_head)

        self.aggression_head = (
            AggressionHead(cfg.num_filters, cfg.aggression_hidden_size)
            if cfg.use_aggression_head else None
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict:
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.tower:
            out = block(out)

        result = {
            "policy_logits": self.policy_head(out),
            "value": self.value_head(out),
        }
        if self.aggression_head is not None:
            result["aggression"] = self.aggression_head(out)
        return result

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience inference helper used by MCTS: returns
        (policy_probs, scalar_value_in[-1,1], aggression_in[-1,1])
        with WDL already collapsed to a scalar expectation
        (P(win) - P(loss)) if use_wdl_head is True.
        """
        self.eval()
        out = self.forward(x)
        policy_probs = F.softmax(out["policy_logits"], dim=-1)

        if self.cfg.use_wdl_head:
            wdl = F.softmax(out["value"], dim=-1)  # (B, 3) = [win, draw, loss]
            value = wdl[:, 0] - wdl[:, 2]
        else:
            value = out["value"].squeeze(-1)

        aggression = out.get("aggression")
        if aggression is None:
            aggression = torch.zeros_like(value)

        return policy_probs, value, aggression


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
