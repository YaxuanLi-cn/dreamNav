import argparse
import json
import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ─── Model (same as training) ────────────────────────────────────────────────

class AffinePredictor(nn.Module):
    """MLP:  (delta_range, delta_heading) -> 2x2 affine matrix (tx=ty=0)."""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 1.0]))

    def forward(self, cmd):
        params = self.net(cmd)
        B = params.shape[0]
        mat = params.view(B, 2, 2)
        zeros = torch.zeros(B, 2, 1, device=params.device, dtype=params.dtype)
        theta = torch.cat([mat, zeros], dim=2)
        return theta


def apply_affine(img, theta):
    grid = F.affine_grid(theta, img.size(), align_corners=False)
    return F.grid_sample(img, grid, align_corners=False,
                         mode='bilinear', padding_mode='zeros')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step2 Test: perturb step1 predictions and refine via affine model')
    parser.add_argument('--step1_json',   type=str, default='../step1/step1_test_seen.json',
                        help='Path to step1 prediction JSON')
    parser.add_argument('--data_dir',     type=str, default='../pairUAV',
                        help='Root data directory (contains tours/)')
    parser.add_argument('--model_path',   type=str, default='step2_model.pth',
                        help='Path to trained step2 model')
    parser.add_argument('--img_size',     type=int, default=256)
    parser.add_argument('--hidden_dim',   type=int, default=256)
    parser.add_argument('--range_delta',  type=float, default=20.0,
                        help='Perturbation magnitude for range')
    parser.add_argument('--heading_delta', type=float, default=10.0,
                        help='Perturbation magnitude for heading')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Max number of samples to test (0 = all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── load model ────────────────────────────────────────────────────
    model = AffinePredictor(hidden_dim=args.hidden_dim).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded model from {args.model_path}')

    # ── load step1 predictions ────────────────────────────────────────
    with open(args.step1_json, 'r') as f:
        step1 = json.load(f)

    pred_headings = step1['pred_deg_num']
    true_headings = step1['true_deg_num']
    pred_ranges   = step1['pred_rag_num']
    true_ranges   = step1['true_rag_num']
    json_paths    = step1['json_path']
    N = len(json_paths)
    if args.max_samples > 0 and N > args.max_samples:
        N = args.max_samples
        pred_headings = pred_headings[:N]
        true_headings = true_headings[:N]
        pred_ranges   = pred_ranges[:N]
        true_ranges   = true_ranges[:N]
        json_paths    = json_paths[:N]
    print(f'Loaded {N} samples from {args.step1_json}')

    # ── image transform ───────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # ── perturbation offsets: 3 x 3 = 9 combinations ─────────────────
    range_offsets   = [-args.range_delta, 0.0, args.range_delta]
    heading_offsets = [-args.heading_delta, 0.0, args.heading_delta]
    perturbations = []
    for dr in range_offsets:
        for dh in heading_offsets:
            perturbations.append((dr, dh))
    num_perturb = len(perturbations)  # 9
    print(f'Perturbations: range_delta={args.range_delta}, heading_delta={args.heading_delta}, '
          f'{num_perturb} combinations')

    # ── iterate over samples ──────────────────────────────────────────
    refined_headings = []
    refined_ranges   = []

    with torch.no_grad():
        for i in range(N):
            # load pair metadata
            jp = json_paths[i]
            # json_path is relative like ../pairUAV/try_test/0000/03_33.json
            # resolve relative to step2 directory or use as-is
            if not os.path.isabs(jp):
                jp_abs = os.path.normpath(os.path.join(os.path.dirname(args.step1_json), jp))
            else:
                jp_abs = jp
            with open(jp_abs, 'r') as f:
                pair = json.load(f)

            # load images
            img_a = Image.open(os.path.join(args.data_dir, 'tours', pair['image_a'])).convert('RGB')
            img_b = Image.open(os.path.join(args.data_dir, 'tours', pair['image_b'])).convert('RGB')
            img_a_t = tf(img_a).unsqueeze(0).to(device)  # [1, 3, H, W]
            img_b_t = tf(img_b).unsqueeze(0).to(device)

            pred_r = pred_ranges[i]
            pred_h = pred_headings[i]

            # build 9 perturbed commands
            cmds = []
            candidate_ranges = []
            candidate_headings = []
            for dr, dh in perturbations:
                r_cand = pred_r + dr
                h_cand = pred_h + dh
                candidate_ranges.append(r_cand)
                candidate_headings.append(h_cand)
                # normalise the same way as training
                cmd = torch.tensor([r_cand / 132.0, h_cand / 180.0], dtype=torch.float32)
                cmds.append(cmd)

            cmds_batch = torch.stack(cmds, dim=0).to(device)  # [9, 2]

            # expand source image to batch of 9
            img_a_batch = img_a_t.expand(num_perturb, -1, -1, -1)  # [9, 3, H, W]

            # get affine matrices and warp
            theta = model(cmds_batch)                      # [9, 2, 3]
            warped = apply_affine(img_a_batch, theta)      # [9, 3, H, W]

            # compare each warped image to target (L1 per sample)
            img_b_batch = img_b_t.expand(num_perturb, -1, -1, -1)
            l1_per_sample = (warped - img_b_batch).abs().mean(dim=[1, 2, 3])  # [9]

            best_idx = l1_per_sample.argmin().item()
            refined_ranges.append(candidate_ranges[best_idx])
            refined_headings.append(candidate_headings[best_idx])

            if (i + 1) % 100 == 0 or i == 0:
                print(f'[{i+1}/{N}] best_perturb=({perturbations[best_idx][0]:+.0f}, '
                      f'{perturbations[best_idx][1]:+.0f})  '
                      f'l1_best={l1_per_sample[best_idx]:.4f}')

    # ─── compute MAE ──────────────────────────────────────────────────
    true_h = np.array(true_headings)
    true_r = np.array(true_ranges)
    pred_h = np.array(pred_headings)
    pred_r = np.array(pred_ranges)
    ref_h  = np.array(refined_headings)
    ref_r  = np.array(refined_ranges)

    def angular_abs_diff(a, b):
        """Smallest absolute angular difference, handling wraparound."""
        diff = np.abs(a - b) % 360.0
        return np.minimum(diff, 360.0 - diff)

    # before refinement
    mae_heading_before = np.mean(angular_abs_diff(pred_h, true_h))
    mae_range_before   = np.mean(np.abs(pred_r - true_r))

    # after refinement
    mae_heading_after = np.mean(angular_abs_diff(ref_h, true_h))
    mae_range_after   = np.mean(np.abs(ref_r - true_r))

    print('\n' + '=' * 60)
    print(f'Before refinement (step1 raw):')
    print(f'  Heading MAE: {mae_heading_before:.4f}')
    print(f'  Range   MAE: {mae_range_before:.4f}')
    print(f'\nAfter refinement (step2 perturb):')
    print(f'  Heading MAE: {mae_heading_after:.4f}')
    print(f'  Range   MAE: {mae_range_after:.4f}')
    print(f'\nImprovement:')
    print(f'  Heading MAE: {mae_heading_before:.4f} -> {mae_heading_after:.4f} '
          f'({"+" if mae_heading_after > mae_heading_before else ""}'
          f'{mae_heading_after - mae_heading_before:.4f})')
    print(f'  Range   MAE: {mae_range_before:.4f} -> {mae_range_after:.4f} '
          f'({"+" if mae_range_after > mae_range_before else ""}'
          f'{mae_range_after - mae_range_before:.4f})')
    print('=' * 60)


if __name__ == '__main__':
    main()
