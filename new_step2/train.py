"""
Training script for new_step2: Residual Refinement with Structure Alignment.

Training strategies (addresses user question D):
═══════════════════════════════════════════════════════════════════════════════

1. Supervised (--mode supervised):
   - L = λ_pose · (λ_h · L_heading + λ_r · L_range)
   - Use when Δ_gt is available. Fastest convergence.

2. Self-supervised (--mode self_supervised):
   - L = λ_struct · L_struct
   - Use when Δ_gt is NOT available. Relies entirely on structure alignment.
   - Risk of local optima → mitigations:
     a) Multi-scale structure loss (coarse-to-fine gradient).
     b) Curriculum: start with large struct_scales, add fine scales later.
     c) Robust loss clamping to avoid outlier gradients.
     d) Separate heading/range: can freeze one and train the other.

3. Joint (--mode joint)  ★ RECOMMENDED:
   - L = λ_pose · L_pose + λ_struct(epoch) · L_struct
   - λ_struct ramps up linearly over struct_warmup_epochs.
   - Pose loss provides strong direct gradient; structure loss adds
     texture-invariant geometric consistency signal.

Avoiding "warp approximation not accurate enough" problems:
  - Multi-scale structure: coarse scales tolerate warp inaccuracy.
  - Robust loss clamping: prevents extreme structure loss from dominating.
  - Curriculum warmup: structure loss starts at 0 and ramps up.
  - Gradient clipping: prevents exploding gradients from warp.
"""

import json
import math
import os
import random
import time
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_args, NORM_RANGE_MAX, NORM_RANGE_MIN
from dataset import Step2Dataset
from models import SoftEdgeExtractor, DifferentiableWarp, ResidualRefiner
from losses import total_loss


# ══════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════

class Summary(Enum):
    NONE = 0
    AVERAGE = 1

class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_struct_weight(epoch, args):
    """Linearly ramp up structure loss weight over warmup epochs."""
    if args.struct_warmup_epochs <= 0:
        return args.lambda_struct
    progress = min(epoch / args.struct_warmup_epochs, 1.0)
    return args.lambda_struct * progress


# ══════════════════════════════════════════════════════════════════════════
#  Evaluation metrics (same as step1 for fair comparison)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(heading_pred_rad, range_pred_m, gt_heading_deg, gt_range_m):
    """Compute heading MAE (deg), range MAE (m), and success rate (<10 m)."""
    gt_heading_rad = torch.deg2rad(gt_heading_deg)

    # heading MAE
    cos_d = (torch.cos(heading_pred_rad) * torch.cos(gt_heading_rad) +
             torch.sin(heading_pred_rad) * torch.sin(gt_heading_rad)).clamp(-1, 1)
    sin_d = (torch.cos(heading_pred_rad) * torch.sin(gt_heading_rad) -
             torch.sin(heading_pred_rad) * torch.cos(gt_heading_rad))
    delta_deg = torch.rad2deg(torch.atan2(sin_d, cos_d))
    heading_mae = delta_deg.abs().mean().item()

    # range MAE
    range_mae = (range_pred_m - gt_range_m).abs().mean().item()

    # success rate: endpoint distance < 10 m
    pred_x = range_pred_m * torch.cos(heading_pred_rad)
    pred_y = range_pred_m * torch.sin(heading_pred_rad)
    true_x = gt_range_m * torch.cos(gt_heading_rad)
    true_y = gt_range_m * torch.sin(gt_heading_rad)
    endpoint_dist = torch.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
    success_rate = (endpoint_dist < 10.0).float().mean().item()

    return heading_mae, range_mae, success_rate


# ══════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model, edge_extractor, warp_module,
    dataloader, optimizer, epoch, device, args,
    test_loader=None, log_path=None,
):
    model.train()
    warp_module.train()

    meters = {k: AverageMeter(k, ':.4f') for k in
              ['total', 'pose', 'heading', 'range', 'struct']}
    m_hmae = AverageMeter('h_mae', ':.2f')
    m_rmae = AverageMeter('r_mae', ':.2f')
    m_sr = AverageMeter('SR', ':.4f')

    struct_w = get_struct_weight(epoch, args)

    t0 = time.time()
    for i, batch in enumerate(dataloader):
        img_c = batch['img_c'].to(device, non_blocking=True)
        img_t = batch['img_t'].to(device, non_blocking=True)
        d0_h = batch['pred_heading_deg'].to(device, non_blocking=True)
        d0_r = batch['pred_range_m'].to(device, non_blocking=True)
        gt_h = batch['gt_heading_deg'].to(device, non_blocking=True)
        gt_r = batch['gt_range_m'].to(device, non_blocking=True)

        # ── forward ───────────────────────────────────────────────────────
        h_ref, r_ref, r_h, r_r = model(img_c, img_t, d0_h, d0_r)

        # ── structure extraction ──────────────────────────────────────────
        with torch.no_grad():
            # edge extractor is fixed (no learnable params); run in no_grad
            # for efficiency. Gradients flow through warp grid, not edge pixels.
            edge_c = edge_extractor(img_c)
            edge_t = edge_extractor(img_t)

        # ── loss ──────────────────────────────────────────────────────────
        loss, loss_dict = total_loss(
            heading_refined_rad=h_ref,
            range_refined_m=r_ref,
            gt_heading_deg=gt_h,
            gt_range_m=gt_r,
            edge_c=edge_c,
            edge_t=edge_t,
            warp_module=warp_module,
            mode=args.mode,
            lambda_pose=args.lambda_pose,
            lambda_heading=args.lambda_heading,
            lambda_range=args.lambda_range,
            lambda_struct=struct_w,
            struct_scales=args.struct_scales,
            edge_threshold=args.edge_threshold,
            dt_bidirectional=args.dt_bidirectional,
            use_chamfer=args.use_chamfer,
            lambda_chamfer=args.lambda_chamfer,
            robust_clamp=args.struct_loss_clamp if args.robust_struct_loss else 0.0,
        )

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # ── metrics ───────────────────────────────────────────────────────
        bs = img_c.size(0)
        for k, v in loss_dict.items():
            if k in meters:
                meters[k].update(v, bs)

        h_mae, r_mae, sr = compute_metrics(h_ref.detach(), r_ref.detach(),
                                            gt_h, gt_r)
        m_hmae.update(h_mae, bs)
        m_rmae.update(r_mae, bs)
        m_sr.update(sr, bs)

        if i % args.print_freq == 0:
            elapsed = time.time() - t0
            print(f'  Epoch [{epoch}][{i}/{len(dataloader)}]  '
                  f'loss={meters["total"].avg:.4f}  '
                  f'pose={meters["pose"].avg:.4f}  '
                  f'struct={meters["struct"].avg:.4f}(w={struct_w:.4f})  '
                  f'h_mae={m_hmae.avg:.2f}°  r_mae={m_rmae.avg:.2f}m  '
                  f'SR={m_sr.avg*100:.1f}%  '
                  f'time={elapsed:.0f}s', flush=True)

        # ── quick test every N iters ──────────────────────────────────────
        if (args.quick_test_freq > 0
                and (i + 1) % args.quick_test_freq == 0
                and test_loader is not None):
            qs = validate(model, edge_extractor, warp_module,
                          test_loader, device, args,
                          max_samples=args.quick_test_samples)
            qline = (
                f'  [QuickTest ep{epoch} iter{i+1}] '
                f'h_mae={qs["heading_mae"]:.2f}° '
                f'r_mae={qs["range_mae"]:.2f}m '
                f'SR={qs["success_rate"]*100:.1f}% | '
                f'step1: h_mae={qs["heading_mae_s1"]:.2f}° '
                f'r_mae={qs["range_mae_s1"]:.2f}m '
                f'SR={qs["success_rate_s1"]*100:.1f}%'
            )
            print(qline, flush=True)
            if log_path:
                with open(log_path, 'a') as f:
                    f.write(qline + '\n')
            # switch back to train mode
            model.train()
            warp_module.train()

    return {
        'loss': meters['total'].avg,
        'heading_mae': m_hmae.avg,
        'range_mae': m_rmae.avg,
        'success_rate': m_sr.avg,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Validation
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, edge_extractor, warp_module, dataloader, device, args,
             max_samples=0):
    """Validate on test set.  If max_samples > 0, stop after that many samples."""
    model.eval()
    warp_module.eval()

    m_hmae = AverageMeter('h_mae', ':.2f')
    m_rmae = AverageMeter('r_mae', ':.2f')
    m_sr = AverageMeter('SR', ':.4f')

    # also compute step1 baseline metrics
    m_hmae_s1 = AverageMeter('h_mae_s1', ':.2f')
    m_rmae_s1 = AverageMeter('r_mae_s1', ':.2f')
    m_sr_s1 = AverageMeter('SR_s1', ':.4f')

    seen = 0
    for batch in dataloader:
        img_c = batch['img_c'].to(device, non_blocking=True)
        img_t = batch['img_t'].to(device, non_blocking=True)
        d0_h = batch['pred_heading_deg'].to(device, non_blocking=True)
        d0_r = batch['pred_range_m'].to(device, non_blocking=True)
        gt_h = batch['gt_heading_deg'].to(device, non_blocking=True)
        gt_r = batch['gt_range_m'].to(device, non_blocking=True)

        h_ref, r_ref, _, _ = model(img_c, img_t, d0_h, d0_r)

        bs = img_c.size(0)

        # refined metrics
        h_mae, r_mae, sr = compute_metrics(h_ref, r_ref, gt_h, gt_r)
        m_hmae.update(h_mae, bs)
        m_rmae.update(r_mae, bs)
        m_sr.update(sr, bs)

        # step1 baseline metrics (for comparison)
        d0_h_rad = torch.deg2rad(d0_h)
        h_mae_s1, r_mae_s1, sr_s1 = compute_metrics(d0_h_rad, d0_r, gt_h, gt_r)
        m_hmae_s1.update(h_mae_s1, bs)
        m_rmae_s1.update(r_mae_s1, bs)
        m_sr_s1.update(sr_s1, bs)

        seen += bs
        if max_samples > 0 and seen >= max_samples:
            break

    return {
        'heading_mae': m_hmae.avg,
        'range_mae': m_rmae.avg,
        'success_rate': m_sr.avg,
        'heading_mae_s1': m_hmae_s1.avg,
        'range_mae_s1': m_rmae_s1.avg,
        'success_rate_s1': m_sr_s1.avg,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Modules ───────────────────────────────────────────────────────────
    model = ResidualRefiner(
        backbone=args.backbone,
        feat_dim=args.feat_dim,
        max_heading_residual_deg=args.max_heading_residual_deg,
        max_range_residual=args.max_range_residual,
    ).to(device)

    edge_extractor = SoftEdgeExtractor(sigma=args.edge_sigma).to(device)
    edge_extractor.eval()  # no learnable params

    warp_module = DifferentiableWarp(
        warp_type=args.warp_type,
        heading_warp_scale=args.heading_warp_scale,
        range_warp_scale=args.range_warp_scale,
        learnable=args.learnable_warp,
    ).to(device)

    total_p, train_p = count_parameters(model)
    print(f'Refiner params: {total_p:,} total, {train_p:,} trainable')
    if args.learnable_warp:
        wp, wtp = count_parameters(warp_module)
        print(f'Warp params: {wp:,} total, {wtp:,} trainable')

    # ── Optimizer ─────────────────────────────────────────────────────────
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': args.lr_backbone},
        {'params': list(model.delta_embed.parameters()) +
                   list(model.head.parameters()), 'lr': args.lr},
    ]
    if args.learnable_warp:
        param_groups.append({'params': warp_module.parameters(), 'lr': args.lr})

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay)

    # ── Scheduler ─────────────────────────────────────────────────────────
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_lr_size, gamma=args.step_lr_gamma)

    # ── warmup scheduler (manual) ─────────────────────────────────────────
    warmup_epochs = args.warmup_epochs

    # ── Data ──────────────────────────────────────────────────────────────
    print('Loading training data ...')
    train_ds = Step2Dataset(
        data_dir=args.data_dir,
        pair_dir=args.train_dir,
        img_size=args.img_size,
        step1_json=args.step1_train_json,
        heading_noise_std=args.heading_noise_std,
        range_noise_std=args.range_noise_std,
        is_train=True,
    )

    print('Loading test data ...')
    test_ds = Step2Dataset(
        data_dir=args.data_dir,
        pair_dir=args.test_dir,
        img_size=args.img_size,
        step1_json=args.step1_test_json,
        heading_noise_std=0.0,
        range_noise_std=0.0,
        is_train=False,
    )

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, drop_last=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_sr = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        warp_module.load_state_dict(ckpt['warp'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_sr = ckpt.get('best_sr', 0.0)
        print(f'Resumed from epoch {start_epoch}, best SR={best_sr*100:.1f}%')

    # ── Training loop ─────────────────────────────────────────────────────
    log_path = os.path.join(args.save_dir, 'output.log')
    open(log_path, 'w').close()

    for epoch in range(start_epoch, args.epochs):
        # warmup LR scaling
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * warmup_factor / max(
                    epoch / warmup_epochs, 1e-8) if epoch > 0 else pg['lr'] * warmup_factor

        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{args.epochs-1}  '
              f'lr={optimizer.param_groups[-1]["lr"]:.2e}  '
              f'struct_w={get_struct_weight(epoch, args):.4f}')
        print(f'{"="*60}')

        train_stats = train_one_epoch(
            model, edge_extractor, warp_module,
            train_loader, optimizer, epoch, device, args,
            test_loader=test_loader, log_path=log_path)

        val_stats = validate(
            model, edge_extractor, warp_module,
            test_loader, device, args)

        if epoch >= warmup_epochs:
            scheduler.step()

        # ── logging ───────────────────────────────────────────────────────
        line = (
            f'Epoch {epoch} | '
            f'Train loss={train_stats["loss"]:.4f} '
            f'h_mae={train_stats["heading_mae"]:.2f}° '
            f'r_mae={train_stats["range_mae"]:.2f}m '
            f'SR={train_stats["success_rate"]*100:.1f}% | '
            f'Val h_mae={val_stats["heading_mae"]:.2f}° '
            f'r_mae={val_stats["range_mae"]:.2f}m '
            f'SR={val_stats["success_rate"]*100:.1f}% | '
            f'Step1 baseline: h_mae={val_stats["heading_mae_s1"]:.2f}° '
            f'r_mae={val_stats["range_mae_s1"]:.2f}m '
            f'SR={val_stats["success_rate_s1"]*100:.1f}%'
        )
        print(f'\n{line}\n')
        with open(log_path, 'a') as f:
            f.write(line + '\n')

        # ── checkpoint ────────────────────────────────────────────────────
        is_best = val_stats['success_rate'] > best_sr
        if is_best:
            best_sr = val_stats['success_rate']

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'warp': warp_module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_stats': val_stats,
            'best_sr': best_sr,
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pt'))
        if is_best:
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))
            print(f'  ★ New best SR: {best_sr*100:.1f}%')

    print(f'\nTraining complete. Best SR: {best_sr*100:.1f}%')
    print(f'Checkpoints saved to {args.save_dir}/')


if __name__ == '__main__':
    main()
