import argparse
import json
import math
import os
import time
import logging
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ========================= Normalization Constants =========================
norm_range_max = 132.0
norm_range_min = -132.0

# ========================= Loss Functions =========================

def angle_loss_cos_sin(pred_xy, target_vec, beta=0.1):
    pred_xy = F.normalize(pred_xy, dim=-1)
    target_vec = F.normalize(target_vec, dim=-1)
    loss = F.smooth_l1_loss(pred_xy, target_vec, beta=beta, reduction='mean')
    return loss


def range_loss(pred_range, target_range, beta=0.1):
    loss = F.smooth_l1_loss(pred_range, target_range, beta=beta, reduction='mean')
    return loss


# ========================= Transforms =========================

def get_transforms(img_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    train_transforms = A.Compose([
        A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return val_transforms, train_transforms


# ========================= Scheduler =========================

def get_scheduler(optimizer, warmup_epochs, total_epochs, step_size, gamma):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(warmup_epochs, 1))
        else:
            return gamma ** ((current_epoch - warmup_epochs) // step_size)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ========================= Dataset =========================

class TourFrameMatchDataset(Dataset):
    """
    Dataset that loads image pairs AND their SuperGlue match displacement field.
    Combines the baseline's dual-image approach with step1's match field approach.
    """

    def __init__(self, json_dir, image_root, match_dir, transforms=None,
                 match_field_size=384, orig_w=640, orig_h=480):
        super().__init__()

        self.image_root = image_root
        self.match_dir = match_dir
        self.transforms = transforms
        self.match_field_size = match_field_size
        self.sx = float(match_field_size) / orig_w
        self.sy = float(match_field_size) / orig_h

        self.json_paths = []
        for g in sorted(os.listdir(json_dir)):
            gdir = os.path.join(json_dir, g)
            if not os.path.isdir(gdir):
                continue
            for f in sorted(os.listdir(gdir)):
                if f.endswith('.json'):
                    self.json_paths.append(os.path.join(gdir, f))

        self.samples = []
        skipped = 0

        for json_path in tqdm(self.json_paths, desc='Loading dataset'):
            with open(json_path, 'r') as fp:
                data = json.load(fp)

            image_a_path = os.path.join(self.image_root, data['image_a'])
            image_b_path = os.path.join(self.image_root, data['image_b'])

            # Derive match npz path
            p = Path(json_path)
            json_id = p.parent.name  # e.g., "0839"
            a_name = Path(data['image_a']).stem  # e.g., "image-03"
            b_name = Path(data['image_b']).stem  # e.g., "image-33"
            npz_path = os.path.join(self.match_dir, json_id, f"{a_name}_{b_name}_matches.npz")

            if not os.path.isfile(npz_path):
                skipped += 1
                continue

            theta_deg = float(data['heading_num'])
            theta_rad = math.radians(theta_deg)
            norm_heading = [math.cos(theta_rad), math.sin(theta_rad)]

            range_num = float(data['range_num'])
            norm_range_val = (range_num - norm_range_min) / (norm_range_max - norm_range_min)

            self.samples.append({
                'image_a_path': image_a_path,
                'image_b_path': image_b_path,
                'npz_path': npz_path,
                'heading': theta_deg,
                'norm_heading': norm_heading,
                'range': range_num,
                'norm_range': norm_range_val,
                'json_path': json_path,
            })

        if skipped > 0:
            print(f'Warning: skipped {skipped} samples due to missing match files')
        print(f'Loaded {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def _build_match_field(self, npz_path):
        """Build a 2-channel displacement field from SuperGlue matches."""
        sz = self.match_field_size
        with np.load(npz_path, allow_pickle=False) as z:
            k0 = z['keypoints0']    # [N, 2] (x, y)
            k1 = z['keypoints1']    # [M, 2]
            m = z['matches']        # [N], target index, <0 means invalid

        valid = (m >= 0) & (m < len(k1))
        src = k0[valid]
        dst = k1[m[valid]]

        # Scale to match_field_size
        x_src = src[:, 0] * self.sx
        y_src = src[:, 1] * self.sy
        x_dst = dst[:, 0] * self.sx
        y_dst = dst[:, 1] * self.sy

        xi = np.clip(x_src, 0, sz - 0.0001).astype(np.int32)
        yi = np.clip(y_src, 0, sz - 0.0001).astype(np.int32)

        dx = (x_dst - x_src).astype(np.float32)
        dy = (y_dst - y_src).astype(np.float32)

        match_field = np.zeros((2, sz, sz), dtype=np.float32)
        match_field[0, yi, xi] = dx
        match_field[1, yi, xi] = dy

        return torch.from_numpy(match_field)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_a = cv2.imread(sample['image_a_path'])
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

        img_b = cv2.imread(sample['image_b_path'])
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img_a = self.transforms(image=img_a)['image']
            img_b = self.transforms(image=img_b)['image']

        match_field = self._build_match_field(sample['npz_path'])

        label_heading = torch.tensor(sample['heading'], dtype=torch.float32)
        label_norm_heading = torch.tensor(sample['norm_heading'], dtype=torch.float32)
        label_range = torch.tensor(sample['range'], dtype=torch.float32)
        label_norm_range = torch.tensor(sample['norm_range'], dtype=torch.float32)

        return (img_a, img_b, match_field,
                label_heading, label_norm_heading,
                sample['json_path'],
                label_range, label_norm_range)


# ========================= timm compatibility =========================

# timm 0.6.x uses different model names than timm >= 0.8
MODEL_NAME_MAP = {
    'convnext_base.fb_in22k_ft_in1k_384': 'convnext_base_384_in22ft1k',
    'convnext_base.fb_in22k_ft_in1k': 'convnext_base_in22ft1k',
}

def resolve_model_name(name):
    """Map new-style timm model names to old-style if needed."""
    if name in MODEL_NAME_MAP:
        try:
            timm.create_model(name, pretrained=False)
            return name
        except RuntimeError:
            return MODEL_NAME_MAP[name]
    return name

def get_data_config(model):
    """Get data config compatible with both old and new timm."""
    try:
        return timm.data.resolve_model_data_config(model)
    except AttributeError:
        from timm.data import resolve_data_config
        return resolve_data_config({}, model=model)


# ========================= Model =========================

class MatchEncoder(nn.Module):
    """Small CNN to encode the 2-channel match displacement field into a feature vector."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.encoder(x)


class EnhancedModel(nn.Module):
    """
    Enhanced model combining:
    - ConvNeXt backbone for dual-image feature extraction (from baseline)
    - Match displacement field encoder (from step1's concept)
    - Multi-layer regressor with proper capacity
    """

    def __init__(self, model_name, pretrained=True, img_size=384, match_feat_dim=256):
        super(EnhancedModel, self).__init__()

        # ConvNeXt backbone (shared for both images)
        resolved_name = resolve_model_name(model_name)
        self.backbone = timm.create_model(resolved_name, pretrained=pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features  # 1024 for convnext_base

        # Match field encoder
        self.match_encoder = MatchEncoder(out_dim=match_feat_dim)

        # Multi-layer regressor: backbone_dim*2 + match_feat_dim -> 3
        total_feat_dim = backbone_dim * 2 + match_feat_dim
        self.regressor = nn.Sequential(
            nn.Linear(total_feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    def get_config(self):
        return get_data_config(self.backbone)

    def forward(self, img_a, img_b, match_field):
        feat_a = self.backbone(img_a)          # (B, backbone_dim)
        feat_b = self.backbone(img_b)          # (B, backbone_dim)
        match_feat = self.match_encoder(match_field)  # (B, match_feat_dim)

        feat = torch.cat([feat_a, feat_b, match_feat], dim=1)
        output = self.regressor(feat)
        return output


# ========================= Validation =========================

def validate(test_loader, model, device, logger):
    model.eval()
    all_heading_errors = []
    all_range_errors = []
    all_success = []

    # Per-sample predictions for saving best results
    now_deg_preds = []
    now_deg_trues = []
    now_rag_preds = []
    now_rag_trues = []
    now_jsons = []

    with torch.no_grad():
        for (img_a, img_b, match_field,
             label_heading, label_norm_heading, json_paths,
             label_range, label_norm_range) in tqdm(test_loader, desc='[Validate]'):

            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)
            match_field = match_field.to(device, non_blocking=True)
            label_heading = label_heading.to(device, non_blocking=True)
            label_norm_heading = label_norm_heading.to(device, non_blocking=True)
            label_range = label_range.to(device, non_blocking=True)

            output = model(img_a, img_b, match_field)

            # Decode range
            pred_range = output[:, 0] * (norm_range_max - norm_range_min) + norm_range_min

            # Decode heading: normalize the cos/sin prediction
            pred_heading_vec = F.normalize(output[:, 1:], dim=-1)

            # Predicted heading in degrees
            pred_heading_deg = torch.rad2deg(torch.atan2(pred_heading_vec[:, 1], pred_heading_vec[:, 0]))

            # Range MAE
            all_range_errors.append(torch.abs(pred_range - label_range))

            # Heading MAE (circular)
            cos_d = (pred_heading_vec * label_norm_heading).sum(dim=-1).clamp(-1.0, 1.0)
            sin_d = (pred_heading_vec[:, 0] * label_norm_heading[:, 1]
                     - pred_heading_vec[:, 1] * label_norm_heading[:, 0])
            delta_rad = torch.atan2(sin_d, cos_d)
            heading_diff = torch.rad2deg(delta_rad).abs()
            all_heading_errors.append(heading_diff)

            # Success rate: Euclidean distance between endpoints < 10m
            true_heading_rad = label_heading * math.pi / 180.0
            pred_heading_rad = torch.atan2(pred_heading_vec[:, 1], pred_heading_vec[:, 0])
            true_x = label_range * torch.cos(true_heading_rad)
            true_y = label_range * torch.sin(true_heading_rad)
            pred_x = pred_range * torch.cos(pred_heading_rad)
            pred_y = pred_range * torch.sin(pred_heading_rad)
            dist = torch.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)
            all_success.append((dist < 10.0).float())

            # Collect per-sample predictions
            now_deg_preds.extend(pred_heading_deg.detach().cpu().tolist())
            now_deg_trues.extend(label_heading.detach().cpu().tolist())
            now_rag_preds.extend(pred_range.detach().cpu().tolist())
            now_rag_trues.extend(label_range.detach().cpu().tolist())
            now_jsons.extend(list(json_paths))

    all_heading_errors = torch.cat(all_heading_errors)
    all_range_errors = torch.cat(all_range_errors)
    all_success = torch.cat(all_success)

    heading_mae = all_heading_errors.mean()
    range_mae = all_range_errors.mean()
    success_rate = all_success.mean() * 100.0

    predictions = {
        'pred_deg_num': now_deg_preds,
        'true_deg_num': now_deg_trues,
        'pred_rag_num': now_rag_preds,
        'true_rag_num': now_rag_trues,
        'json_path': now_jsons,
    }

    return range_mae, heading_mae, success_rate, predictions


# ========================= Training =========================

best_metric = 1e9
best_range_mae = 1e9
best_heading_mae = 1e9
best_success_rate = 0.0


def train_one_epoch(train_loader, test_loader, model, optimizer, epoch, device, logger):
    global best_metric, best_range_mae, best_heading_mae, best_success_rate

    model.train()

    running_loss = 0.0
    running_angle_loss = 0.0
    running_range_loss = 0.0
    num_batches = 0

    for (img_a, img_b, match_field,
         label_heading, label_norm_heading, _,
         label_range, label_norm_range) in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):

        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        match_field = match_field.to(device, non_blocking=True)
        label_norm_heading = label_norm_heading.to(device, non_blocking=True)
        label_norm_range = label_norm_range.to(device, non_blocking=True)

        output = model(img_a, img_b, match_field)

        pred_heading = output[:, 1:]
        pred_range = output[:, 0]

        loss_angle = angle_loss_cos_sin(pred_heading, label_norm_heading)
        loss_range = range_loss(pred_range, label_norm_range)
        loss = loss_angle + loss_range

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        running_angle_loss += loss_angle.item()
        running_range_loss += loss_range.item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    avg_angle = running_angle_loss / max(num_batches, 1)
    avg_range = running_range_loss / max(num_batches, 1)

    logger.info(f'Epoch {epoch} | Train Loss: {avg_loss:.4f} (angle: {avg_angle:.4f}, range: {avg_range:.4f})')

    # Validate
    range_mae, heading_mae, success_rate, predictions = validate(test_loader, model, device, logger)

    msg = (f'Epoch {epoch} | Range MAE: {range_mae.item():.2f} | '
           f'Heading MAE: {heading_mae.item():.2f} | Success Rate: {success_rate.item():.2f}%')
    logger.info(msg)

    metric = range_mae.item() + heading_mae.item()
    if metric < best_metric:
        best_metric = metric
        best_range_mae = range_mae.item()
        best_heading_mae = heading_mae.item()
        best_success_rate = success_rate.item()

        # Save best predictions to JSON (like step1.py)
        pred_json_path = 'enhanced_seen.json'
        with open(pred_json_path, 'w') as f:
            json.dump(predictions, f)
        logger.info(f'New best! Saved predictions to {pred_json_path}')

    best_msg = (f'Best so far | Range MAE: {best_range_mae:.2f} | '
                f'Heading MAE: {best_heading_mae:.2f} | Success Rate: {best_success_rate:.2f}%')
    logger.info(best_msg)

    return metric


# ========================= Main =========================

parser = argparse.ArgumentParser(description='Enhanced training with ConvNeXt + SuperGlue matches')
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--lr_backbone', default=5e-5, type=float,
                    help='Learning rate for ConvNeXt backbone')
parser.add_argument('--lr_match_encoder', default=1e-3, type=float,
                    help='Learning rate for match field encoder')
parser.add_argument('--lr_regressor', default=1e-3, type=float,
                    help='Learning rate for regressor head')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=1e-4, type=float, dest='weight_decay')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--warmup_epochs', default=1, type=int)
parser.add_argument('--image_root', type=str, required=True,
                    help='Path to tours image root (e.g., pairUAV/tours)')
parser.add_argument('--train_dir', type=str, required=True,
                    help='Path to training set JSON directory')
parser.add_argument('--test_dir', type=str, required=True,
                    help='Path to test set JSON directory')
parser.add_argument('--match_dir', type=str, required=True,
                    help='Path to SuperGlue match output directory (matches_data)')
parser.add_argument('--output_file', type=str, default='test_results_enhanced.log')
parser.add_argument('--model_name', type=str, default='convnext_base.fb_in22k_ft_in1k_384')
parser.add_argument('--img_size', type=int, default=384)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--checkpoint', type=str, default='',
                    help='Path to pretrained Sample4Geo backbone checkpoint (optional)')
parser.add_argument('--match_feat_dim', type=int, default=256,
                    help='Match encoder output feature dimension')
parser.add_argument('--save_dir', type=str, default='checkpoints_enhanced',
                    help='Directory to save model checkpoints')


def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Logger
    logger = logging.getLogger('enhanced_train')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(args.output_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Build model
    logger.info(f'Building model: {args.model_name}')
    model = EnhancedModel(
        model_name=args.model_name,
        pretrained=True,
        img_size=args.img_size,
        match_feat_dim=args.match_feat_dim,
    )

    # Load pretrained Sample4Geo backbone checkpoint if provided
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info(f'Loading backbone checkpoint: {args.checkpoint}')
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        # Handle potential key prefix differences
        # Sample4Geo saves TimmModel state_dict with keys like "model.xxx" and "logit_scale"
        # Our backbone is directly a timm model, so we need keys like "xxx" (without "model." prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[len('model.'):]] = v
            elif k == 'logit_scale':
                continue  # skip contrastive learning scale
            else:
                new_state_dict[k] = v
        missing, unexpected = model.backbone.load_state_dict(new_state_dict, strict=False)
        logger.info(f'Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}')
        if missing:
            logger.info(f'Missing keys (first 5): {missing[:5]}')
        if unexpected:
            logger.info(f'Unexpected keys (first 5): {unexpected[:5]}')
    else:
        logger.info('No checkpoint provided, using ImageNet pretrained backbone')

    model.to(device)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')

    # Image transforms
    img_size = (args.img_size, args.img_size)
    data_config = get_data_config(model.backbone)
    mean = list(data_config["mean"])
    std = list(data_config["std"])
    val_transforms, train_transforms = get_transforms(img_size, mean=mean, std=std)

    # Datasets
    logger.info('Loading training dataset...')
    train_dataset = TourFrameMatchDataset(
        args.train_dir, args.image_root, args.match_dir,
        transforms=train_transforms, match_field_size=args.img_size,
    )
    logger.info('Loading test dataset...')
    test_dataset = TourFrameMatchDataset(
        args.test_dir, args.image_root, args.match_dir,
        transforms=val_transforms, match_field_size=args.img_size,
    )

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Optimizer with different learning rates for each component
    param_groups = [
        {'params': list(model.backbone.parameters()), 'lr': args.lr_backbone},
        {'params': list(model.match_encoder.parameters()), 'lr': args.lr_match_encoder},
        {'params': list(model.regressor.parameters()), 'lr': args.lr_regressor},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        step_size=30,
        gamma=0.1,
    )

    logger.info(f'Training for {args.epochs} epochs')
    logger.info(f'LR backbone: {args.lr_backbone}, LR match_encoder: {args.lr_match_encoder}, '
                f'LR regressor: {args.lr_regressor}')
    logger.info(f'Batch size: {args.batch_size}, Optimizer: AdamW')

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        logger.info(f'=== Epoch {epoch} ===')
        metric = train_one_epoch(
            train_dataloader, test_dataloader, model, optimizer, epoch, device, logger
        )
        scheduler.step()

        # Save checkpoint every epoch
        ckpt_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f'Saved checkpoint: {ckpt_path}')

    logger.info('Training complete.')
    logger.info(f'Best | Range MAE: {best_range_mae:.2f} | '
                f'Heading MAE: {best_heading_mae:.2f} | Success Rate: {best_success_rate:.2f}%')


if __name__ == '__main__':
    main()
