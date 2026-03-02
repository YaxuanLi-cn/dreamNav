"""
Standalone evaluation script for new_step2.

Loads a trained checkpoint and evaluates on test set, comparing
refined predictions against step1 baseline.

Also saves per-sample predictions to JSON for downstream analysis.
"""

import json
import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_args, NORM_RANGE_MAX, NORM_RANGE_MIN
from dataset import Step2Dataset
from models import SoftEdgeExtractor, DifferentiableWarp, ResidualRefiner


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Full evaluation with per-sample results."""
    model.eval()

    all_pred_heading_deg = []
    all_pred_range_m = []
    all_gt_heading_deg = []
    all_gt_range_m = []
    all_s1_heading_deg = []
    all_s1_range_m = []

    for batch in tqdm(dataloader, desc='Evaluating'):
        img_c = batch['img_c'].to(device, non_blocking=True)
        img_t = batch['img_t'].to(device, non_blocking=True)
        d0_h = batch['pred_heading_deg'].to(device, non_blocking=True)
        d0_r = batch['pred_range_m'].to(device, non_blocking=True)
        gt_h = batch['gt_heading_deg']
        gt_r = batch['gt_range_m']

        h_ref, r_ref, _, _ = model(img_c, img_t, d0_h, d0_r)

        # convert heading to degrees
        h_ref_deg = torch.rad2deg(h_ref).cpu()

        all_pred_heading_deg.extend(h_ref_deg.tolist())
        all_pred_range_m.extend(r_ref.cpu().tolist())
        all_gt_heading_deg.extend(gt_h.tolist())
        all_gt_range_m.extend(gt_r.tolist())
        all_s1_heading_deg.extend(d0_h.cpu().tolist())
        all_s1_range_m.extend(d0_r.cpu().tolist())

    # ── Compute metrics ───────────────────────────────────────────────────
    pred_h = torch.tensor(all_pred_heading_deg)
    pred_r = torch.tensor(all_pred_range_m)
    gt_h = torch.tensor(all_gt_heading_deg)
    gt_r = torch.tensor(all_gt_range_m)
    s1_h = torch.tensor(all_s1_heading_deg)
    s1_r = torch.tensor(all_s1_range_m)

    # Refined metrics
    ref_stats = _compute_full_metrics(
        torch.deg2rad(pred_h), pred_r, gt_h, gt_r)

    # Step1 baseline metrics
    s1_stats = _compute_full_metrics(
        torch.deg2rad(s1_h), s1_r, gt_h, gt_r)

    return {
        'refined': ref_stats,
        'step1': s1_stats,
        'predictions': {
            'pred_heading_deg': all_pred_heading_deg,
            'pred_range_m': all_pred_range_m,
            'gt_heading_deg': all_gt_heading_deg,
            'gt_range_m': all_gt_range_m,
            's1_heading_deg': all_s1_heading_deg,
            's1_range_m': all_s1_range_m,
        },
    }


def _compute_full_metrics(heading_rad, range_m, gt_heading_deg, gt_range_m):
    gt_heading_rad = torch.deg2rad(gt_heading_deg)

    # heading error
    cos_d = (torch.cos(heading_rad) * torch.cos(gt_heading_rad) +
             torch.sin(heading_rad) * torch.sin(gt_heading_rad)).clamp(-1, 1)
    sin_d = (torch.cos(heading_rad) * torch.sin(gt_heading_rad) -
             torch.sin(heading_rad) * torch.cos(gt_heading_rad))
    delta_deg = torch.rad2deg(torch.atan2(sin_d, cos_d))
    heading_mae = delta_deg.abs().mean().item()
    heading_mse = (delta_deg ** 2).mean().item()

    # range error
    range_err = range_m - gt_range_m
    range_mae = range_err.abs().mean().item()
    range_mse = (range_err ** 2).mean().item()

    # endpoint distance
    pred_x = range_m * torch.cos(heading_rad)
    pred_y = range_m * torch.sin(heading_rad)
    true_x = gt_range_m * torch.cos(gt_heading_rad)
    true_y = gt_range_m * torch.sin(gt_heading_rad)
    endpoint_dist = torch.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
    success_rate = (endpoint_dist < 10.0).float().mean().item()
    mean_endpoint = endpoint_dist.mean().item()

    return {
        'heading_mae': heading_mae,
        'heading_mse': heading_mse,
        'range_mae': range_mae,
        'range_mse': range_mse,
        'success_rate': success_rate,
        'mean_endpoint_dist': mean_endpoint,
    }


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.save_dir, 'best.pt')
    if args.resume:
        ckpt_path = args.resume
    if not os.path.isfile(ckpt_path):
        print(f'Checkpoint not found: {ckpt_path}')
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get('args', {})
    print(f'Loading checkpoint from {ckpt_path} (epoch {ckpt.get("epoch", "?")})')

    model = ResidualRefiner(
        backbone=saved_args.get('backbone', args.backbone),
        feat_dim=saved_args.get('feat_dim', args.feat_dim),
        max_heading_residual_deg=saved_args.get(
            'max_heading_residual_deg', args.max_heading_residual_deg),
        max_range_residual=saved_args.get(
            'max_range_residual', args.max_range_residual),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # ── Data ──────────────────────────────────────────────────────────────
    test_ds = Step2Dataset(
        data_dir=args.data_dir,
        pair_dir=args.test_dir,
        img_size=args.img_size,
        step1_json=args.step1_test_json,
        is_train=False,
    )
    num_workers = min(args.num_workers, os.cpu_count() or 1)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = evaluate(model, test_loader, device)

    # ── Print results ─────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('EVALUATION RESULTS')
    print('=' * 70)

    for tag in ['step1', 'refined']:
        s = results[tag]
        print(f'\n  {tag.upper()}:')
        print(f'    Heading  MAE={s["heading_mae"]:.2f}°  MSE={s["heading_mse"]:.2f}')
        print(f'    Range    MAE={s["range_mae"]:.2f}m  MSE={s["range_mse"]:.2f}')
        print(f'    Success Rate (<10m): {s["success_rate"]*100:.2f}%')
        print(f'    Mean Endpoint Dist:  {s["mean_endpoint_dist"]:.2f}m')

    # improvement
    h_imp = results['step1']['heading_mae'] - results['refined']['heading_mae']
    r_imp = results['step1']['range_mae'] - results['refined']['range_mae']
    sr_imp = results['refined']['success_rate'] - results['step1']['success_rate']
    print(f'\n  IMPROVEMENT over step1:')
    print(f'    Heading MAE: {h_imp:+.2f}° ({h_imp/max(results["step1"]["heading_mae"],1e-8)*100:+.1f}%)')
    print(f'    Range MAE:   {r_imp:+.2f}m ({r_imp/max(results["step1"]["range_mae"],1e-8)*100:+.1f}%)')
    print(f'    Success Rate: {sr_imp*100:+.2f}pp')
    print('=' * 70)

    # ── Save results ──────────────────────────────────────────────────────
    out_dir = args.save_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'eval_results.json'), 'w') as f:
        json.dump({
            'refined': results['refined'],
            'step1': results['step1'],
        }, f, indent=2)

    with open(os.path.join(out_dir, 'predictions.json'), 'w') as f:
        json.dump(results['predictions'], f)

    print(f'\nResults saved to {out_dir}/')


if __name__ == '__main__':
    main()
