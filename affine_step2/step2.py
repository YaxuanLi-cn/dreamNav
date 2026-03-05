import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AffineDataset(Dataset):
    """Loads image pairs with delta range / heading for affine prediction."""

    def __init__(self, data_dir, json_dir, img_size=256, max_samples=None):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size

        self.json_paths = []
        for g in sorted(os.listdir(json_dir)):
            gp = os.path.join(json_dir, g)
            if not os.path.isdir(gp):
                continue
            for f in sorted(os.listdir(gp)):
                if f.endswith('.json'):
                    self.json_paths.append(os.path.join(gp, f))
                    if max_samples and len(self.json_paths) >= max_samples:
                        break
            if max_samples and len(self.json_paths) >= max_samples:
                break

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        with open(self.json_paths[idx], 'r') as f:
            data = json.load(f)

        img_a = Image.open(os.path.join(self.data_dir, 'tours', data['image_a'])).convert('RGB')
        img_b = Image.open(os.path.join(self.data_dir, 'tours', data['image_b'])).convert('RGB')

        img_a = self.transform(img_a)
        img_b = self.transform(img_b)

        delta_range = float(data['range_num']) / 132.0      # normalise to ~[-1, 1]
        delta_heading = float(data['heading_num']) / 180.0   # normalise to  [-1, 1]

        cmd = torch.tensor([delta_range, delta_heading], dtype=torch.float32)
        return cmd, img_a, img_b


# ─── Model ────────────────────────────────────────────────────────────────────

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
            nn.Linear(hidden_dim, 4),          # a, b, c, d
        )
        # initialise last layer so that output ≈ identity [[1,0],[0,1]]
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 1.0]))

    def forward(self, cmd):
        """
        cmd   : [B, 2]
        return: [B, 2, 3]  affine matrix  [[a,b,0],[c,d,0]]
        """
        params = self.net(cmd)                             # [B, 4]
        B = params.shape[0]
        mat = params.view(B, 2, 2)
        zeros = torch.zeros(B, 2, 1, device=params.device, dtype=params.dtype)
        theta = torch.cat([mat, zeros], dim=2)             # [B, 2, 3]
        return theta


def apply_affine(img, theta):
    """Warp *img* with affine matrix *theta* ([B,2,3])."""
    grid = F.affine_grid(theta, img.size(), align_corners=False)
    return F.grid_sample(img, grid, align_corners=False,
                         mode='bilinear', padding_mode='zeros')


# ─── Metrics ──────────────────────────────────────────────────────────────────

def batch_psnr(pred, target):
    """Per-sample PSNR averaged over the batch (images in [0,1])."""
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])      # [B]
    psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))
    return psnr.mean().item()


# ─── Visualisation ────────────────────────────────────────────────────────────

def save_vis(img_a, img_b, warped, vis_dir, tag, max_save=16):
    """Save [source | GT target | predicted] side-by-side images."""
    os.makedirs(vis_dir, exist_ok=True)
    n = min(img_a.size(0), max_save)
    for j in range(n):
        # concat along width: source | GT | predicted
        row = torch.cat([img_a[j], img_b[j], warped[j].clamp(0, 1)], dim=2)  # [3, H, 3W]
        save_image(row, os.path.join(vis_dir, f'{tag}_{j:04d}.png'))


# ─── Train / Validate ─────────────────────────────────────────────────────────

def validate(loader, model, device, vis_dir=None, tag=''):
    """Evaluate on test set.  If vis_dir is set, save visualisations."""
    model.eval()
    l1_sum, mse_sum, psnr_sum = 0.0, 0.0, 0.0
    count = 0
    vis_saved = 0

    with torch.no_grad():
        for cmd, img_a, img_b in loader:
            cmd   = cmd.to(device,   non_blocking=True)
            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)

            theta  = model(cmd)
            warped = apply_affine(img_a, theta)

            bs = cmd.size(0)
            l1_sum  += F.l1_loss(warped, img_b).item()  * bs
            mse_sum += F.mse_loss(warped, img_b).item() * bs
            psnr_sum += batch_psnr(warped, img_b)        * bs
            count += bs

            # save vis for first batch only
            if vis_dir is not None and vis_saved == 0:
                save_vis(img_a.cpu(), img_b.cpu(), warped.cpu(),
                         vis_dir, tag, max_save=16)
                vis_saved = 1

    avg_l1   = l1_sum   / count
    avg_mse  = mse_sum  / count
    avg_psnr = psnr_sum / count
    return avg_l1, avg_mse, avg_psnr


def main():
    parser = argparse.ArgumentParser(description='Step2: Affine view-change prediction')
    parser.add_argument('--data_dir',          type=str,   default='../pairUAV')
    parser.add_argument('--train_path',        type=str,   default='../pairUAV/train')
    parser.add_argument('--test_path',         type=str,   default='../pairUAV/test')
    parser.add_argument('--test_max_samples',  type=int,   default=1000)
    parser.add_argument('--img_size',          type=int,   default=256)
    parser.add_argument('--hidden_dim',        type=int,   default=256)
    parser.add_argument('--batch_size',        type=int,   default=256)
    parser.add_argument('--num_workers',       type=int,   default=8)
    parser.add_argument('--epochs',            type=int,   default=5)
    parser.add_argument('--lr',                type=float, default=1e-3)
    parser.add_argument('--print_freq',        type=int,   default=100)
    parser.add_argument('--vis_freq',          type=int,   default=500)
    parser.add_argument('--vis_dir',           type=str,   default='vis_output')
    parser.add_argument('--save_path',         type=str,   default='step2_model.pth')
    args = parser.parse_args()

    open('output.log', 'w').close()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── data ──────────────────────────────────────────────────────────
    print('Loading datasets …')
    train_ds = AffineDataset(args.data_dir, args.train_path,
                             img_size=args.img_size)
    test_ds  = AffineDataset(args.data_dir, args.test_path,
                             img_size=args.img_size,
                             max_samples=args.test_max_samples)
    print(f'Train samples: {len(train_ds)},  Test samples: {len(test_ds)}')

    nw = min(args.num_workers, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    # ── model ─────────────────────────────────────────────────────────
    model = AffinePredictor(hidden_dim=args.hidden_dim).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_loss = float('inf')
    global_iter = 0

    # ── training loop ─────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_count    = 0
        t0 = time.time()

        for i, (cmd, img_a, img_b) in enumerate(train_loader):
            cmd   = cmd.to(device,   non_blocking=True)
            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)

            theta  = model(cmd)
            warped = apply_affine(img_a, theta)
            loss   = F.l1_loss(warped, img_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = cmd.size(0)
            train_loss_sum += loss.item() * bs
            train_count    += bs
            global_iter    += 1

            if i % args.print_freq == 0:
                avg = train_loss_sum / train_count
                print(f'Epoch [{epoch}][{i}/{len(train_loader)}]  '
                      f'loss {loss.item():.6f}  avg {avg:.6f}')

            # ── periodic vis + test every vis_freq iters ──────────
            if global_iter % args.vis_freq == 0:
                tag = f'iter{global_iter:06d}'
                print(f'\n--- Vis & Test @ iter {global_iter} ---')
                test_l1, test_mse, test_psnr = validate(
                    test_loader, model, device,
                    vis_dir=args.vis_dir, tag=tag)
                print(f'    Test L1: {test_l1:.6f} | MSE: {test_mse:.6f} | '
                      f'PSNR: {test_psnr:.2f} dB\n')
                with open('output.log', 'a') as f:
                    f.write(f'{tag} | Test L1: {test_l1:.6f} | '
                            f'MSE: {test_mse:.6f} | PSNR: {test_psnr:.2f} dB\n')
                if test_l1 < best_test_loss:
                    best_test_loss = test_l1
                    raw = model.module if hasattr(model, 'module') else model
                    torch.save(raw.state_dict(), args.save_path)
                    print(f'  >> Saved best model (test L1 = {test_l1:.6f})\n')
                model.train()

        scheduler.step()
        train_loss = train_loss_sum / train_count
        train_time = time.time() - t0

        # ── end-of-epoch evaluate ─────────────────────────────────
        tag = f'epoch{epoch}'
        test_l1, test_mse, test_psnr = validate(
            test_loader, model, device,
            vis_dir=args.vis_dir, tag=tag)

        log = (f'Epoch {epoch} | Train L1: {train_loss:.6f} | '
               f'Test L1: {test_l1:.6f} | Test MSE: {test_mse:.6f} | '
               f'Test PSNR: {test_psnr:.2f} dB | '
               f'Time: {train_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}')

        print(f'\n===== {log} =====\n')
        with open('output.log', 'a') as f:
            f.write(log + '\n')

        if test_l1 < best_test_loss:
            best_test_loss = test_l1
            raw = model.module if hasattr(model, 'module') else model
            torch.save(raw.state_dict(), args.save_path)
            print(f'  >> Saved best model (test L1 = {test_l1:.6f})\n')

    print(f'Training finished.  Best test L1: {best_test_loss:.6f}')


if __name__ == '__main__':
    main()
