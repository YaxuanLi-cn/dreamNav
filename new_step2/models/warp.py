"""
Differentiable warp module for 2-DoF (range, heading).

Addresses user question A: how to design a differentiable warp without K/D.
══════════════════════════════════════════════════════════════════════════════

Two warp approximations are provided:

┌─────────────────────────────────────────────────────────────────────────────┐
│ Option 1  —  affine_st  (Scale + Translation)  ★ RECOMMENDED              │
│                                                                             │
│   heading (rad) → horizontal translation  tx = tanh(α · θ)                │
│   range   (norm) → isotropic scale         s = exp(β · r_norm)            │
│                                                                             │
│   Affine matrix:  [[s, 0, tx],                                             │
│                    [0, s,  0]]                                              │
│                                                                             │
│   Physical intuition:                                                       │
│     - Drone heading change ≈ lateral scene shift in image plane.           │
│     - Range change ≈ zoom (closer = bigger, farther = smaller).            │
│   Stability:                                                                │
│     - tanh bounds tx to [-1,1] → no out-of-bound grid.                    │
│     - exp(·) is always > 0, smooth gradient, symmetric in log-space.      │
│     - Works well for heading ∈ [-π, π] and normalised range ∈ [-1, 1].   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Option 2  —  affine_rs  (Rotation + Scale)                                 │
│                                                                             │
│   heading (rad) → in-plane rotation  θ_rot = α · heading_rad              │
│   range   (norm) → isotropic scale    s    = exp(β · r_norm)              │
│                                                                             │
│   Affine matrix:  [[s·cos θ_rot, -s·sin θ_rot, 0],                        │
│                    [s·sin θ_rot,  s·cos θ_rot, 0]]                         │
│                                                                             │
│   Physical intuition:                                                       │
│     - Yaw rotation ≈ image-plane rotation (only exact for nadir view).    │
│   Stability:                                                                │
│     - sin/cos are bounded → inherently stable.                             │
│     - Less accurate for oblique drone views; heading→rotation is a        │
│       rougher approximation than heading→translation for forward cameras. │
└─────────────────────────────────────────────────────────────────────────────┘

Recommendation:
  affine_st is more stable and physically appropriate for forward-facing drone
  cameras.  affine_rs can be useful for top-down/nadir views where yaw maps
  naturally to in-plane rotation.

Parameter normalisation:
  - heading: input in **radians** (from atan2 of cos/sin prediction).
  - range:   normalised to [-1, 1] via  r_norm = range_m / NORM_RANGE_MAX.
  - α (heading_warp_scale): default 0.5; tanh(0.5 * π) ≈ 1.0, so full
    heading range maps to full horizontal shift.
  - β (range_warp_scale): default 0.3; exp(±0.3) ∈ [0.74, 1.35], giving
    ±26–35 % zoom for extreme ranges.

Gradient stability:
  - tanh and exp have smooth, non-zero gradients everywhere.
  - F.grid_sample with bilinear mode + zero-padding provides smooth gradients
    through the spatial transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NORM_RANGE_MAX


class DifferentiableWarp(nn.Module):
    """Warp a single-channel feature map according to 2-DoF action (heading, range)."""

    def __init__(
        self,
        warp_type: str = 'affine_st',
        heading_warp_scale: float = 0.5,
        range_warp_scale: float = 0.3,
        learnable: bool = False,
    ):
        super().__init__()
        self.warp_type = warp_type

        if learnable:
            self.heading_scale = nn.Parameter(torch.tensor(heading_warp_scale))
            self.range_scale = nn.Parameter(torch.tensor(range_warp_scale))
        else:
            self.register_buffer('heading_scale',
                                 torch.tensor(heading_warp_scale))
            self.register_buffer('range_scale',
                                 torch.tensor(range_warp_scale))

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        feat: torch.Tensor,
        heading_rad: torch.Tensor,
        range_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat:        [B, C, H, W]  feature/edge map to warp.
            heading_rad: [B]           heading in radians.
            range_m:     [B]           range in metres.
        Returns:
            warped:      [B, C, H, W]  warped feature map.
        """
        B = feat.shape[0]

        # ── normalise range to [-1, 1] ────────────────────────────────────
        r_norm = range_m / NORM_RANGE_MAX  # [-1, 1]

        if self.warp_type == 'affine_st':
            theta = self._build_affine_st(heading_rad, r_norm, B, feat.device)
        elif self.warp_type == 'affine_rs':
            theta = self._build_affine_rs(heading_rad, r_norm, B, feat.device)
        else:
            raise ValueError(f'Unknown warp_type: {self.warp_type}')

        grid = F.affine_grid(theta, feat.size(), align_corners=False)
        warped = F.grid_sample(
            feat, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        return warped

    # ── Option 1: Scale + Translation ─────────────────────────────────────
    def _build_affine_st(self, heading_rad, r_norm, B, device):
        tx = torch.tanh(self.heading_scale * heading_rad)   # [B], ∈ (-1, 1)
        s = torch.exp(self.range_scale * r_norm)             # [B], > 0

        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = s      # scale x
        theta[:, 1, 1] = s      # scale y
        theta[:, 0, 2] = tx     # translate x
        # ty = 0 (no vertical shift for 2-DoF heading+range)
        return theta

    # ── Option 2: Rotation + Scale ────────────────────────────────────────
    def _build_affine_rs(self, heading_rad, r_norm, B, device):
        rot = self.heading_scale * heading_rad   # [B]
        s = torch.exp(self.range_scale * r_norm)  # [B]

        cos_r = torch.cos(rot)
        sin_r = torch.sin(rot)

        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] =  s * cos_r
        theta[:, 0, 1] = -s * sin_r
        theta[:, 1, 0] =  s * sin_r
        theta[:, 1, 1] =  s * cos_r
        # no translation
        return theta
