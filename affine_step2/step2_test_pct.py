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
    parser = argparse.ArgumentParser(
        description='Step2 Test: perturb step1 predictions by percentage and refine via affine model')
    parser.add_argument('--step1_json',   type=str, default='../step1/step1_test_seen.json',
                        help='Path to step1 prediction JSON')
    parser.add_argument('--data_dir',     type=str, default='../pairUAV',
                        help='Root data directory (contains tours/)')
    parser.add_argument('--model_path',   type=str, default='step2_model.pth',
                        help='Path to trained step2 model')
    parser.add_argument('--img_size',     type=int, default=256)
    parser.add_argument('--hidden_dim',   type=int, default=256)
    parser.add_argument('--range_pct',    type=float, default=5.0,
                        help='Perturbation percentage for range (e.g. 5 means ±5%%)')
    parser.add_argument('--heading_pct',  type=float, default=5.0,
                        help='Perturbation percentage for heading (e.g. 5 means ±5%%)')
    parser.add_argument('--max_samples',  type=int, default=1000,
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

    # ── perturbation ratios: 3 x 3 = 9 combinations ──────────────────
    range_ratios   = [-(args.range_pct / 100.0), 0.0, (args.range_pct / 100.0)]
    heading_ratios = [-(args.heading_pct / 100.0), 0.0, (args.heading_pct / 100.0)]
    ratio_combos = []
    for rr in range_ratios:
        for hr in heading_ratios:
            ratio_combos.append((rr, hr))
    num_perturb = len(ratio_combos)  # 9
    print(f'Perturbations: range_pct=±{args.range_pct}%, heading_pct=±{args.heading_pct}%, '
          f'{num_perturb} combinations')

    # ── iterate over samples ──────────────────────────────────────────
    refined_headings = []
    refined_ranges   = []

    with torch.no_grad():
        for i in range(N):
            # load pair metadata
            jp = json_paths[i]
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

            # build 9 perturbed commands (percentage-based)
            cmds = []
            candidate_ranges = []
            candidate_headings = []
            for rr, hr in ratio_combos:
                r_cand = pred_r * (1.0 + rr)
                h_cand = pred_h * (1.0 + hr)
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
                rr_best, hr_best = ratio_combos[best_idx]
                print(f'[{i+1}/{N}] pred_r={pred_r:.1f} pred_h={pred_h:.1f}  '
                      f'best_pct=({rr_best*100:+.1f}%, {hr_best*100:+.1f}%)  '
                      f'delta=({pred_r*rr_best:+.2f}, {pred_h*hr_best:+.2f})  '
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
    print(f'\nAfter refinement (step2 perturb by percentage):')
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
