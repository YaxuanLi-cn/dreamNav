"""
Residual Refinement Network  f2.

Architecture
────────────
    I_c ─→ ┐
            ├─ shared ResNet-18 encoder ──→ feat_c (512-d)
    I_t ─→ ┘                               feat_t (512-d)

    Δ0 = (heading_deg, range_m) ─→ delta_embed MLP ──→ d (64-d)

    concat(feat_c, feat_t, feat_c − feat_t, d) ──→ fusion MLP ──→ r = (r_θ, r_range)

    Δ_refined:
        θ_refined   = θ0 + max_heading_rad · tanh(r_θ)
        range_refined = range0 + max_range · tanh(r_range)

The tanh clipping guarantees:
  - heading residual  ∈ [-max_heading_rad, +max_heading_rad]
  - range   residual  ∈ [-max_range,       +max_range]
  which prevents the refinement from diverging far from step1 estimates.
"""

import math

import torch
import torch.nn as nn
import torchvision.models as models

from config import NORM_RANGE_MAX, NORM_RANGE_MIN


class ResidualRefiner(nn.Module):
    def __init__(
        self,
        backbone: str = 'resnet18',
        feat_dim: int = 512,
        max_heading_residual_deg: float = 45.0,
        max_range_residual: float = 40.0,
    ):
        super().__init__()

        self.max_heading_residual_rad = math.radians(max_heading_residual_deg)
        self.max_range_residual = max_range_residual

        # ── Shared image encoder ──────────────────────────────────────────
        if backbone == 'resnet18':
            base = models.resnet18(pretrained=True)
        elif backbone == 'resnet34':
            base = models.resnet34(pretrained=True)
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        # Remove classification head; keep up to avgpool
        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,           # output: [B, feat_dim, 1, 1]
        )

        # ── Δ0 embedding ─────────────────────────────────────────────────
        # Input: (cos θ0, sin θ0, norm_range0) → 64-d
        self.delta_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # ── Fusion + residual head ────────────────────────────────────────
        # feat_c (512) + feat_t (512) + diff (512) + delta_embed (64) = 1600
        fuse_dim = feat_dim * 3 + 64
        self.head = nn.Sequential(
            nn.Linear(fuse_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),       # (raw_r_heading, raw_r_range)
        )

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        img_c: torch.Tensor,
        img_t: torch.Tensor,
        delta0_heading_deg: torch.Tensor,
        delta0_range_m: torch.Tensor,
    ):
        """
        Args:
            img_c:              [B, 3, H, W]  current-view image.
            img_t:              [B, 3, H, W]  target-view image.
            delta0_heading_deg: [B]           step1 heading prediction (degrees).
            delta0_range_m:     [B]           step1 range prediction (metres).
        Returns:
            heading_refined_rad: [B]  refined heading in radians.
            range_refined_m:     [B]  refined range in metres.
            r_heading_rad:       [B]  heading residual in radians.
            r_range_m:           [B]  range residual in metres.
        """
        # ── encode images ─────────────────────────────────────────────────
        feat_c = self.encoder(img_c).flatten(1)   # [B, feat_dim]
        feat_t = self.encoder(img_t).flatten(1)   # [B, feat_dim]

        # ── encode Δ0 ─────────────────────────────────────────────────────
        theta0_rad = torch.deg2rad(delta0_heading_deg)
        cos0 = torch.cos(theta0_rad)
        sin0 = torch.sin(theta0_rad)
        norm_range0 = (delta0_range_m - NORM_RANGE_MIN) / (NORM_RANGE_MAX - NORM_RANGE_MIN)

        delta0_vec = torch.stack([cos0, sin0, norm_range0], dim=-1)  # [B, 3]
        d = self.delta_embed(delta0_vec)                              # [B, 64]

        # ── fuse ──────────────────────────────────────────────────────────
        diff = feat_c - feat_t
        fused = torch.cat([feat_c, feat_t, diff, d], dim=-1)  # [B, 1600]

        raw = self.head(fused)  # [B, 2]

        # ── bounded residual ──────────────────────────────────────────────
        r_heading_rad = self.max_heading_residual_rad * torch.tanh(raw[:, 0])
        r_range_m = self.max_range_residual * torch.tanh(raw[:, 1])

        # ── refined prediction ────────────────────────────────────────────
        heading_refined_rad = theta0_rad + r_heading_rad
        range_refined_m = delta0_range_m + r_range_m

        return heading_refined_rad, range_refined_m, r_heading_rad, r_range_m
