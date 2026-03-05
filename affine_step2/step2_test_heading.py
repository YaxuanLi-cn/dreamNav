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


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_candidate_values(path):
    """Load numeric values from a text file (one per line)."""
    values = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    return np.array(values)


def angular_distance(a, b):
    """Smallest absolute angular difference, handling wraparound at 360."""
    diff = np.abs(a - b) % 360.0
    return np.minimum(diff, 360.0 - diff)


def find_closest_n(values, target, n=9):
    """Find the n closest values in *values* to *target* (angular distance)."""
    dists = angular_distance(values, target)
    indices = np.argsort(dists)[:n]
    return values[indices]


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Step2 Test: refine heading only using closest candidates from range_num.txt')
    parser.add_argument('--step1_json',      type=str, default='../step1/step1_test_seen.json',
                        help='Path to step1 prediction JSON')
    parser.add_argument('--data_dir',        type=str, default='../pairUAV',
                        help='Root data directory (contains tours/)')
    parser.add_argument('--model_path',      type=str, default='step2_model.pth',
                        help='Path to trained step2 model')
    parser.add_argument('--candidate_file',  type=str, default='range_num.txt',
                        help='File containing candidate heading values (one per line)')
    parser.add_argument('--img_size',        type=int, default=256)
    parser.add_argument('--hidden_dim',      type=int, default=256)
    parser.add_argument('--num_candidates',  type=int, default=9,
                        help='Number of closest heading candidates to try')
    parser.add_argument('--max_samples',     type=int, default=1000,
                        help='Max number of samples to test (0 = all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── load candidate heading values from file ────────────────────────
    candidate_values = load_candidate_values(args.candidate_file)
    print(f'Loaded {len(candidate_values)} candidate values from {args.candidate_file}')

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

    num_cand = args.num_candidates

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

            # find 9 closest heading candidates from candidate_values
            heading_candidates = find_closest_n(candidate_values, pred_h, n=num_cand)

            # build commands: range stays fixed, heading varies
            cmds = []
            for h_cand in heading_candidates:
                cmd = torch.tensor([pred_r / 132.0, h_cand / 180.0], dtype=torch.float32)
                cmds.append(cmd)

            cmds_batch = torch.stack(cmds, dim=0).to(device)  # [num_cand, 2]

            # expand source image to batch
            img_a_batch = img_a_t.expand(num_cand, -1, -1, -1)  # [num_cand, 3, H, W]

            # get affine matrices and warp
            theta = model(cmds_batch)                      # [num_cand, 2, 3]
            warped = apply_affine(img_a_batch, theta)      # [num_cand, 3, H, W]

            # compare each warped image to target (L1 per sample)
            img_b_batch = img_b_t.expand(num_cand, -1, -1, -1)
            l1_per_sample = (warped - img_b_batch).abs().mean(dim=[1, 2, 3])  # [num_cand]

            best_idx = l1_per_sample.argmin().item()
            refined_ranges.append(pred_r)  # range unchanged
            refined_headings.append(heading_candidates[best_idx])

            if (i + 1) % 100 == 0 or i == 0:
                print(f'[{i+1}/{N}] pred_h={pred_h:.1f} -> refined_h={heading_candidates[best_idx]:.1f}  '
                      f'candidates={np.sort(heading_candidates).tolist()}  '
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
    print(f'\nAfter refinement (heading only, {num_cand} closest candidates):')
    print(f'  Heading MAE: {mae_heading_after:.4f}')
    print(f'  Range   MAE: {mae_range_after:.4f}')
    print(f'\nImprovement:')
    print(f'  Heading MAE: {mae_heading_before:.4f} -> {mae_heading_after:.4f} '
          f'({"+" if mae_heading_after > mae_heading_before else ""}'
          f'{mae_heading_after - mae_heading_before:.4f})')
    print(f'  Range   MAE: unchanged')
    print('=' * 60)


if __name__ == '__main__':
    main()
