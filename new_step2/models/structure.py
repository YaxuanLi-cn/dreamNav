"""
Soft-edge structure extraction S(·).

Design rationale (addresses user question B):
─────────────────────────────────────────────
Why soft-edge + DT loss is the most practical first choice:
  1. Fully differentiable (Sobel = fixed-weight conv2d).
  2. No pretrained model / extra dependency.
  3. Captures geometry (building outlines, road edges) while ignoring texture.
  4. Smooth [0,1] output → stable gradients through grid_sample in warp.
  5. Distance Transform on target edges provides wide basin of attraction
     (gradient even when warped edges are far from target edges).

Alternatives considered but NOT used here:
  - Canny: non-differentiable thresholding (could precompute, but loses grad).
  - Line-segment detector (LSD): not differentiable, sparse output.
  - Semantic boundary: requires pretrained segmentation → heavy, domain-shift risk.

Pipeline:
  RGB → grayscale → Gaussian blur (σ) → Sobel Gx, Gy
  → magnitude = sqrt(Gx² + Gy²) → normalise to [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SoftEdgeExtractor(nn.Module):
    """Differentiable soft-edge extractor based on Sobel filters."""

    def __init__(self, sigma: float = 1.5):
        super().__init__()
        self.sigma = sigma

        # ── Sobel kernels (registered as buffers, not parameters) ─────────
        sobel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]

        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # ── Gaussian kernel for pre-smoothing ─────────────────────────────
        if sigma > 0:
            kernel = self._make_gaussian_kernel(sigma)
            self.register_buffer('gauss_kernel', kernel)
        else:
            self.gauss_kernel = None

    @staticmethod
    def _make_gaussian_kernel(sigma: float, kernel_size: int = 0) -> torch.Tensor:
        if kernel_size == 0:
            kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d.unsqueeze(0).unsqueeze(0)  # [1,1,K,K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB image (any normalisation).
        Returns:
            edge: [B, 1, H, W] soft-edge map in [0, 1].
        """
        # ── RGB → grayscale ───────────────────────────────────────────────
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # ── Gaussian smoothing ────────────────────────────────────────────
        if self.gauss_kernel is not None:
            pad = self.gauss_kernel.shape[-1] // 2
            gray = F.pad(gray, [pad] * 4, mode='reflect')
            gray = F.conv2d(gray, self.gauss_kernel)

        # ── Sobel gradients ───────────────────────────────────────────────
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)

        # ── Gradient magnitude ────────────────────────────────────────────
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # ── Per-sample normalisation to [0, 1] ───────────────────────────
        B = edge.shape[0]
        flat = edge.view(B, -1)
        edge_min = flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        edge_max = flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        edge = (edge - edge_min) / (edge_max - edge_min + 1e-8)

        return edge
