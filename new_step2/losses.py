"""
Loss functions for new_step2.

Addresses user questions C and D.
══════════════════════════════════════════════════════════════════════════════

C) Structure Consistency Losses
────────────────────────────────

1. Distance Transform (DT) Loss  ★ PRIMARY
   ─────────────────────────────
   Given:
     E_t  = binarise(S(I_t), τ)          # target binary edge
     DT_t = distance_transform_edt(1−E_t) # Euclidean DT of background
     DT_t = DT_t / max(DT_t)             # normalise to [0, 1]

   Forward DT loss:
     L_dt_fwd = mean( S_warped · DT_t )
       → penalises warped-edge pixels that are far from any target edge.
       → gradient flows: L → S_warped → grid_sample → grid → Δ → r.

   Backward DT loss (optional, bidirectional):
     DT_warped = DT( binarise(S_warped) )   # computed w/o gradient
     L_dt_bwd  = mean( S_t · DT_warped )
       → penalises target-edge pixels far from any warped edge.

   Bidirectional:
     L_dt = 0.5 · (L_dt_fwd + L_dt_bwd)

   Why DT works well:
     - Wide basin of attraction: even when edges are far apart, the loss
       has non-zero gradient (unlike pixel-wise L1 on edges).
     - Smooth, sub-pixel gradients via bilinear grid_sample.

2. Chamfer Loss (optional)
   ───────────────────────
   Extract top-K edge point coordinates from both maps, compute
   bidirectional nearest-neighbour distance:
     L_chamfer = mean(min_j ‖p_i − q_j‖₂) + mean(min_i ‖p_i − q_j‖₂)

   Complexity: O(K²) per sample — can be expensive; keep K ≤ 512.
   More robust to sparse edges but noisier gradient than DT.

D) Pose Supervision Loss
─────────────────────────
  L_heading = smooth_L1( [cos θ_pred, sin θ_pred], [cos θ_gt, sin θ_gt] )
  L_range   = smooth_L1( norm_range_pred, norm_range_gt )
  L_pose    = λ_h · L_heading + λ_r · L_range

  Joint training:
    L_total = λ_pose · L_pose + λ_struct(epoch) · L_struct

  where λ_struct ramps up linearly during struct_warmup_epochs to avoid
  early-stage structure-loss noise overwhelming the pose signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

from config import NORM_RANGE_MAX, NORM_RANGE_MIN


# ══════════════════════════════════════════════════════════════════════════
#  Distance Transform Loss
# ══════════════════════════════════════════════════════════════════════════

def compute_dt_map(edge_binary_np: np.ndarray) -> np.ndarray:
    """Compute normalised Euclidean Distance Transform of background.

    Args:
        edge_binary_np: [H, W] binary edge map (1 = edge, 0 = background).
    Returns:
        dt: [H, W] normalised DT in [0, 1].
    """
    background = 1.0 - edge_binary_np.astype(np.float64)
    dt = distance_transform_edt(background)
    dt_max = dt.max()
    if dt_max > 0:
        dt = dt / dt_max
    return dt.astype(np.float32)


def dt_loss_forward(
    warped_edge: torch.Tensor,
    target_edge: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Forward DT loss: penalise warped edges far from target edges.

    Args:
        warped_edge:  [B, 1, H, W] soft-edge of warped source (grad needed).
        target_edge:  [B, 1, H, W] soft-edge of target (no grad needed).
        threshold:    binarisation threshold for target edges.
    Returns:
        scalar loss.
    """
    B = warped_edge.shape[0]
    device = warped_edge.device

    with torch.no_grad():
        target_binary = (target_edge > threshold).float()
        dt_maps = []
        for b in range(B):
            edge_np = target_binary[b, 0].cpu().numpy()
            dt = compute_dt_map(edge_np)
            dt_maps.append(torch.from_numpy(dt))
        dt_target = torch.stack(dt_maps, dim=0).unsqueeze(1).to(device)  # [B,1,H,W]

    # weighted mean: warped edge intensity × distance-to-nearest-target-edge
    loss = (warped_edge * dt_target).mean()
    return loss


def dt_loss_backward(
    warped_edge: torch.Tensor,
    target_edge: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Backward DT loss: penalise target edges far from warped edges.

    Note: gradient does NOT flow through this term (DT of warped is non-diff).
    This term only provides an additional learning signal when combined with
    the forward term.
    """
    B = warped_edge.shape[0]
    device = warped_edge.device

    with torch.no_grad():
        warped_binary = (warped_edge > threshold).float()
        dt_maps = []
        for b in range(B):
            edge_np = warped_binary[b, 0].cpu().numpy()
            dt = compute_dt_map(edge_np)
            dt_maps.append(torch.from_numpy(dt))
        dt_warped = torch.stack(dt_maps, dim=0).unsqueeze(1).to(device)

    # target_edge provides gradient here (through edge extractor, if needed)
    loss = (target_edge * dt_warped).mean()
    return loss


def dt_loss(
    warped_edge: torch.Tensor,
    target_edge: torch.Tensor,
    threshold: float = 0.3,
    bidirectional: bool = False,
) -> torch.Tensor:
    """Combined DT loss (forward, optionally + backward)."""
    l_fwd = dt_loss_forward(warped_edge, target_edge, threshold)
    if bidirectional:
        l_bwd = dt_loss_backward(warped_edge, target_edge, threshold)
        return 0.5 * (l_fwd + l_bwd)
    return l_fwd


# ══════════════════════════════════════════════════════════════════════════
#  Chamfer Loss (optional)
# ══════════════════════════════════════════════════════════════════════════

def chamfer_loss(
    warped_edge: torch.Tensor,
    target_edge: torch.Tensor,
    threshold: float = 0.3,
    max_points: int = 512,
) -> torch.Tensor:
    """Bidirectional Chamfer distance between edge point sets.

    Complexity: O(B · K²) where K = max_points.
    More robust to sparse edges, but noisier gradient than DT loss.

    Args:
        warped_edge: [B, 1, H, W]
        target_edge: [B, 1, H, W]
        threshold:   binarisation threshold.
        max_points:  max edge points to sample per image.
    Returns:
        scalar loss.
    """
    B, _, H, W = warped_edge.shape
    device = warped_edge.device
    total_loss = torch.tensor(0.0, device=device)

    # create normalised coordinate grid [H, W, 2] in [-1, 1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij',
    )
    coords = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

    for b in range(B):
        # extract edge point coordinates
        w_mask = (warped_edge[b, 0] > threshold)
        t_mask = (target_edge[b, 0] > threshold)

        w_pts = coords[w_mask]  # [Nw, 2]
        t_pts = coords[t_mask]  # [Nt, 2]

        if w_pts.shape[0] == 0 or t_pts.shape[0] == 0:
            continue

        # subsample if too many points
        if w_pts.shape[0] > max_points:
            idx = torch.randperm(w_pts.shape[0], device=device)[:max_points]
            w_pts = w_pts[idx]
        if t_pts.shape[0] > max_points:
            idx = torch.randperm(t_pts.shape[0], device=device)[:max_points]
            t_pts = t_pts[idx]

        # pairwise distances [Nw, Nt]
        dist = torch.cdist(w_pts.unsqueeze(0), t_pts.unsqueeze(0)).squeeze(0)

        # bidirectional nearest-neighbour
        l_w2t = dist.min(dim=1).values.mean()
        l_t2w = dist.min(dim=0).values.mean()
        total_loss = total_loss + 0.5 * (l_w2t + l_t2w)

    return total_loss / max(B, 1)


# ══════════════════════════════════════════════════════════════════════════
#  Multi-scale Structure Alignment Loss
# ══════════════════════════════════════════════════════════════════════════

def multiscale_structure_loss(
    edge_c: torch.Tensor,
    edge_t: torch.Tensor,
    warp_module,
    heading_rad: torch.Tensor,
    range_m: torch.Tensor,
    scales: list,
    edge_threshold: float = 0.3,
    bidirectional: bool = False,
    use_chamfer: bool = False,
    lambda_chamfer: float = 0.05,
    robust_clamp: float = 0.0,
) -> torch.Tensor:
    """Compute structure alignment loss at multiple scales.

    Args:
        edge_c:       [B, 1, H, W] soft-edge of current view.
        edge_t:       [B, 1, H, W] soft-edge of target view.
        warp_module:  DifferentiableWarp instance.
        heading_rad:  [B] refined heading in radians.
        range_m:      [B] refined range in metres.
        scales:       list of float (e.g. [1.0, 0.5, 0.25]).
        edge_threshold, bidirectional, use_chamfer, lambda_chamfer: loss options.
        robust_clamp: if > 0, clamp per-scale loss to this value.
    Returns:
        scalar loss.
    """
    total = torch.tensor(0.0, device=edge_c.device)

    for s in scales:
        if s < 1.0:
            size = (int(edge_c.shape[2] * s), int(edge_c.shape[3] * s))
            ec = F.interpolate(edge_c, size=size, mode='bilinear',
                               align_corners=False)
            et = F.interpolate(edge_t, size=size, mode='bilinear',
                               align_corners=False)
        else:
            ec, et = edge_c, edge_t

        warped = warp_module(ec, heading_rad, range_m)

        l_dt = dt_loss(warped, et, edge_threshold, bidirectional)

        if use_chamfer:
            l_ch = chamfer_loss(warped, et, edge_threshold)
            l_scale = l_dt + lambda_chamfer * l_ch
        else:
            l_scale = l_dt

        if robust_clamp > 0:
            l_scale = torch.clamp(l_scale, max=robust_clamp)

        total = total + l_scale

    return total / len(scales)


# ══════════════════════════════════════════════════════════════════════════
#  Pose Supervision Losses (same formulation as step1)
# ══════════════════════════════════════════════════════════════════════════

def heading_loss(pred_rad: torch.Tensor, gt_deg: torch.Tensor,
                 beta: float = 0.1) -> torch.Tensor:
    """Smooth-L1 loss on (cos, sin) heading representation."""
    gt_rad = torch.deg2rad(gt_deg)
    pred_cs = torch.stack([torch.cos(pred_rad), torch.sin(pred_rad)], dim=-1)
    gt_cs = torch.stack([torch.cos(gt_rad), torch.sin(gt_rad)], dim=-1)
    pred_cs = F.normalize(pred_cs, dim=-1)
    gt_cs = F.normalize(gt_cs, dim=-1)
    return F.smooth_l1_loss(pred_cs, gt_cs, beta=beta)


def range_loss(pred_m: torch.Tensor, gt_m: torch.Tensor,
               beta: float = 0.1) -> torch.Tensor:
    """Smooth-L1 loss on normalised range."""
    pred_norm = (pred_m - NORM_RANGE_MIN) / (NORM_RANGE_MAX - NORM_RANGE_MIN)
    gt_norm = (gt_m - NORM_RANGE_MIN) / (NORM_RANGE_MAX - NORM_RANGE_MIN)
    return F.smooth_l1_loss(pred_norm, gt_norm, beta=beta)


# ══════════════════════════════════════════════════════════════════════════
#  Total Loss
# ══════════════════════════════════════════════════════════════════════════

def total_loss(
    # predictions
    heading_refined_rad: torch.Tensor,
    range_refined_m: torch.Tensor,
    # ground truth
    gt_heading_deg: torch.Tensor,
    gt_range_m: torch.Tensor,
    # structure inputs
    edge_c: torch.Tensor,
    edge_t: torch.Tensor,
    warp_module,
    # config
    mode: str = 'joint',
    lambda_pose: float = 1.0,
    lambda_heading: float = 1.0,
    lambda_range: float = 1.0,
    lambda_struct: float = 0.1,
    struct_scales: list = None,
    edge_threshold: float = 0.3,
    dt_bidirectional: bool = False,
    use_chamfer: bool = False,
    lambda_chamfer: float = 0.05,
    robust_clamp: float = 0.0,
):
    """Compute total training loss.

    Returns:
        loss_total, dict of sub-losses for logging.
    """
    if struct_scales is None:
        struct_scales = [1.0, 0.5, 0.25]

    losses = {}

    # ── Pose supervision ──────────────────────────────────────────────────
    if mode in ('supervised', 'joint'):
        l_h = heading_loss(heading_refined_rad, gt_heading_deg)
        l_r = range_loss(range_refined_m, gt_range_m)
        l_pose = lambda_heading * l_h + lambda_range * l_r
        losses['heading'] = l_h.item()
        losses['range'] = l_r.item()
        losses['pose'] = l_pose.item()
    else:
        l_pose = torch.tensor(0.0, device=heading_refined_rad.device)

    # ── Structure alignment ───────────────────────────────────────────────
    if mode in ('self_supervised', 'joint'):
        l_struct = multiscale_structure_loss(
            edge_c, edge_t, warp_module,
            heading_refined_rad, range_refined_m,
            struct_scales, edge_threshold,
            dt_bidirectional, use_chamfer, lambda_chamfer, robust_clamp,
        )
        losses['struct'] = l_struct.item()
    else:
        l_struct = torch.tensor(0.0, device=heading_refined_rad.device)

    # ── combine ───────────────────────────────────────────────────────────
    l_total = lambda_pose * l_pose + lambda_struct * l_struct
    losses['total'] = l_total.item()

    return l_total, losses
