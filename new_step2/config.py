"""
Configuration for new_step2: Residual Refinement with Structure Alignment.

Key design choices:
  - heading represented as angle (degrees/radians), residual is angular offset
  - range in meters, residual is additive offset
  - warp: affine_st (Scale+Translation, recommended) or affine_rs (Rotation+Scale)
  - structure: differentiable Sobel soft-edge
  - loss: DT loss (primary) + optional Chamfer
"""

import argparse

# === Data normalization constants (matching step1) ===
NORM_RANGE_MAX = 132.0
NORM_RANGE_MIN = -132.0


def get_args():
    parser = argparse.ArgumentParser(
        description='New Step2: Residual Refinement with Structure Alignment')

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument('--data_dir', type=str,
                        default='/root/autodl-tmp/dreamnav')
    parser.add_argument('--train_dir', type=str,
                        default='/root/autodl-tmp/dreamnav/train/')
    parser.add_argument('--test_dir', type=str,
                        default='/root/autodl-tmp/dreamnav/test/')
    parser.add_argument('--step1_test_json', type=str,
                        default='/root/dreamNav/step1/step1_seen.json',
                        help='Step1 predictions on test set')
    parser.add_argument('--step1_train_json', type=str, default='',
                        help='Step1 predictions on train set. '
                             'If empty, simulate with noise.')

    # ── Image ─────────────────────────────────────────────────────────────
    parser.add_argument('--img_size', type=int, default=224)

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'])
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='Backbone output dim (512 for resnet18/34)')

    # ── Residual bounds ───────────────────────────────────────────────────
    parser.add_argument('--max_heading_residual_deg', type=float, default=45.0,
                        help='Max heading correction in degrees')
    parser.add_argument('--max_range_residual', type=float, default=40.0,
                        help='Max range correction in meters')

    # ── Structure extraction ──────────────────────────────────────────────
    parser.add_argument('--edge_sigma', type=float, default=1.5,
                        help='Gaussian sigma before Sobel')
    parser.add_argument('--edge_threshold', type=float, default=0.3,
                        help='Binarization threshold for DT computation')

    # ── Warp ──────────────────────────────────────────────────────────────
    parser.add_argument('--warp_type', type=str, default='affine_st',
                        choices=['affine_st', 'affine_rs'],
                        help='affine_st=Scale+Translation (recommended), '
                             'affine_rs=Rotation+Scale')
    parser.add_argument('--heading_warp_scale', type=float, default=0.5,
                        help='Maps heading (rad) to translation/rotation magnitude')
    parser.add_argument('--range_warp_scale', type=float, default=0.3,
                        help='Maps normalised range to log-scale factor')
    parser.add_argument('--learnable_warp', action='store_true', default=False,
                        help='Make warp scale factors learnable')

    # ── Loss weights ──────────────────────────────────────────────────────
    parser.add_argument('--lambda_pose', type=float, default=1.0)
    parser.add_argument('--lambda_struct', type=float, default=0.1,
                        help='Weight for structure alignment auxiliary loss')
    parser.add_argument('--lambda_range', type=float, default=1.0)
    parser.add_argument('--lambda_heading', type=float, default=1.0)
    parser.add_argument('--use_chamfer', action='store_true', default=False,
                        help='Add Chamfer loss alongside DT loss')
    parser.add_argument('--lambda_chamfer', type=float, default=0.05)
    parser.add_argument('--dt_bidirectional', action='store_true', default=False,
                        help='Use bidirectional DT loss')
    parser.add_argument('--robust_struct_loss', action='store_true', default=True,
                        help='Clamp structure loss to avoid outlier gradients')
    parser.add_argument('--struct_loss_clamp', type=float, default=5.0)

    # ── Multi-scale structure ─────────────────────────────────────────────
    parser.add_argument('--struct_scales', type=float, nargs='+',
                        default=[1.0, 0.5, 0.25],
                        help='Scales for multi-scale structure alignment')

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for refinement head')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate for pretrained backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--quick_test_freq', type=int, default=1000,
                        help='Run quick validation every N training iters (0=disable)')
    parser.add_argument('--quick_test_samples', type=int, default=2000,
                        help='Max test samples used in quick validation')

    # ── Noise simulation (training w/o step1 predictions) ─────────────────
    parser.add_argument('--heading_noise_std', type=float, default=40.0,
                        help='Std of heading noise (deg) for simulating step1')
    parser.add_argument('--range_noise_std', type=float, default=30.0,
                        help='Std of range noise (m) for simulating step1')

    # ── Training mode ─────────────────────────────────────────────────────
    parser.add_argument('--mode', type=str, default='joint',
                        choices=['supervised', 'self_supervised', 'joint'],
                        help='supervised: pose loss only; '
                             'self_supervised: structure loss only; '
                             'joint: both')

    # ── Curriculum / scheduling ───────────────────────────────────────────
    parser.add_argument('--struct_warmup_epochs', type=int, default=5,
                        help='Linearly ramp lambda_struct from 0 over this many epochs')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step'],
                        help='LR scheduler type')
    parser.add_argument('--step_lr_gamma', type=float, default=0.5)
    parser.add_argument('--step_lr_size', type=int, default=10)

    # ── Checkpoint / output ───────────────────────────────────────────────
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()
