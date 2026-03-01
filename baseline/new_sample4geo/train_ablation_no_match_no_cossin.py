import argparse
import json
import math
import os
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


norm_range_max = 132.0
norm_range_min = -132.0


def angle_loss_normalized(pred_norm_angle, target_norm_angle, beta=0.1):
    return F.smooth_l1_loss(pred_norm_angle, target_norm_angle, beta=beta, reduction='mean')


def range_loss(pred_range, target_range, beta=0.1):
    return F.smooth_l1_loss(pred_range, target_range, beta=beta, reduction='mean')


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


def get_scheduler(optimizer, warmup_epochs, total_epochs, step_size, gamma):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(warmup_epochs, 1))
        return gamma ** ((current_epoch - warmup_epochs) // step_size)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TourFramePairDataset(Dataset):
    def __init__(self, json_dir, image_root, transforms=None):
        super().__init__()
        self.image_root = image_root
        self.transforms = transforms

        json_paths = []
        for g in sorted(os.listdir(json_dir)):
            gdir = os.path.join(json_dir, g)
            if not os.path.isdir(gdir):
                continue
            for f in sorted(os.listdir(gdir)):
                if f.endswith('.json'):
                    json_paths.append(os.path.join(gdir, f))
        self.json_paths = json_paths

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        json_path = self.json_paths[idx]
        with open(json_path, 'r') as fp:
            data = json.load(fp)

        image_a_path = os.path.join(self.image_root, data['image_a'])
        image_b_path = os.path.join(self.image_root, data['image_b'])

        img_a = cv2.imread(image_a_path)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

        img_b = cv2.imread(image_b_path)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img_a = self.transforms(image=img_a)['image']
            img_b = self.transforms(image=img_b)['image']

        theta_deg = float(data['heading_num'])
        label_norm_angle = torch.tensor(theta_deg / 180.0, dtype=torch.float32)
        label_heading_deg = torch.tensor(theta_deg, dtype=torch.float32)

        range_num = float(data['range_num'])
        norm_range_val = (range_num - norm_range_min) / (norm_range_max - norm_range_min)

        label_range = torch.tensor(range_num, dtype=torch.float32)
        label_norm_range = torch.tensor(norm_range_val, dtype=torch.float32)

        return (
            img_a,
            img_b,
            label_heading_deg,
            label_norm_angle,
            json_path,
            label_range,
            label_norm_range,
        )


MODEL_NAME_MAP = {
    'convnext_base.fb_in22k_ft_in1k_384': 'convnext_base_384_in22ft1k',
    'convnext_base.fb_in22k_ft_in1k': 'convnext_base_in22ft1k',
}


def resolve_model_name(name):
    if name in MODEL_NAME_MAP:
        try:
            timm.create_model(name, pretrained=False)
            return name
        except RuntimeError:
            return MODEL_NAME_MAP[name]
    return name


def get_data_config(model):
    try:
        return timm.data.resolve_model_data_config(model)
    except AttributeError:
        from timm.data import resolve_data_config

        return resolve_data_config({}, model=model)


class EnhancedAblationModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        resolved_name = resolve_model_name(model_name)
        self.backbone = timm.create_model(resolved_name, pretrained=pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features

        total_feat_dim = backbone_dim * 2
        self.regressor = nn.Sequential(
            nn.Linear(total_feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, img_a, img_b):
        feat_a = self.backbone(img_a)
        feat_b = self.backbone(img_b)
        feat = torch.cat([feat_a, feat_b], dim=1)
        return self.regressor(feat)


def validate(test_loader, model, device, logger):
    model.eval()
    all_heading_errors = []
    all_range_errors = []
    all_success = []

    now_deg_preds = []
    now_deg_trues = []
    now_rag_preds = []
    now_rag_trues = []
    now_jsons = []

    with torch.no_grad():
        for (img_a, img_b,
             label_heading_deg, label_norm_angle, json_paths,
             label_range, _) in tqdm(test_loader, desc='[Validate]'):

            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)
            label_heading_deg = label_heading_deg.to(device, non_blocking=True)
            label_norm_angle = label_norm_angle.to(device, non_blocking=True)
            label_range = label_range.to(device, non_blocking=True)

            output = model(img_a, img_b)

            pred_norm_angle = output[:, 0]
            pred_range = output[:, 1] * (norm_range_max - norm_range_min) + norm_range_min

            pred_heading_deg = pred_norm_angle * 180.0
            delta_deg = pred_heading_deg - label_heading_deg
            delta_deg = (delta_deg + 180.0) % 360.0 - 180.0
            heading_diff = delta_deg.abs()

            all_heading_errors.append(heading_diff)
            all_range_errors.append(torch.abs(pred_range - label_range))

            pred_rad = torch.deg2rad(pred_heading_deg)
            true_rad = torch.deg2rad(label_heading_deg)
            pred_x = pred_range * torch.cos(pred_rad)
            pred_y = pred_range * torch.sin(pred_rad)
            true_x = label_range * torch.cos(true_rad)
            true_y = label_range * torch.sin(true_rad)
            dist = torch.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)
            all_success.append((dist < 10.0).float())

            now_deg_preds.extend(pred_heading_deg.detach().cpu().tolist())
            now_deg_trues.extend(label_heading_deg.detach().cpu().tolist())
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

    for (img_a, img_b,
         _, label_norm_angle, _,
         _, label_norm_range) in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):

        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        label_norm_angle = label_norm_angle.to(device, non_blocking=True)
        label_norm_range = label_norm_range.to(device, non_blocking=True)

        output = model(img_a, img_b)
        pred_norm_angle = output[:, 0]
        pred_norm_range = output[:, 1]

        loss_angle = angle_loss_normalized(pred_norm_angle, label_norm_angle)
        loss_range = range_loss(pred_norm_range, label_norm_range)
        loss = loss_angle + loss_range

        optimizer.zero_grad()
        loss.backward()
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

        pred_json_path = 'enhanced_ablation_no_match_no_cossin_seen.json'
        with open(pred_json_path, 'w') as f:
            json.dump(predictions, f)
        logger.info(f'New best! Saved predictions to {pred_json_path}')

    best_msg = (f'Best so far | Range MAE: {best_range_mae:.2f} | '
                f'Heading MAE: {best_heading_mae:.2f} | Success Rate: {best_success_rate:.2f}%')
    logger.info(best_msg)

    return metric


parser = argparse.ArgumentParser(description='Ablation: remove SuperGlue matches and remove cos/sin heading representation')
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--lr_backbone', default=5e-5, type=float)
parser.add_argument('--lr_regressor', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=1e-4, type=float, dest='weight_decay')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--warmup_epochs', default=1, type=int)
parser.add_argument('--image_root', type=str, required=True)
parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--output_file', type=str, default='test_results_enhanced_ablation_no_match_no_cossin.log')
parser.add_argument('--model_name', type=str, default='convnext_base.fb_in22k_ft_in1k_384')
parser.add_argument('--img_size', type=int, default=384)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--save_dir', type=str, default='checkpoints_enhanced_ablation_no_match_no_cossin')


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger = logging.getLogger('enhanced_ablation_train')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(args.output_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f'Building model: {args.model_name}')
    model = EnhancedAblationModel(model_name=args.model_name, pretrained=True)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info(f'Loading backbone checkpoint: {args.checkpoint}')
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[len('model.'):]] = v
            elif k == 'logit_scale':
                continue
            else:
                new_state_dict[k] = v
        missing, unexpected = model.backbone.load_state_dict(new_state_dict, strict=False)
        logger.info(f'Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}')
    else:
        logger.info('No checkpoint provided, using ImageNet pretrained backbone')

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')

    img_size = (args.img_size, args.img_size)
    data_config = get_data_config(model.backbone)
    mean = list(data_config['mean'])
    std = list(data_config['std'])
    val_transforms, train_transforms = get_transforms(img_size, mean=mean, std=std)

    logger.info('Loading training dataset...')
    train_dataset = TourFramePairDataset(args.train_dir, args.image_root, transforms=train_transforms)
    logger.info('Loading test dataset...')
    test_dataset = TourFramePairDataset(args.test_dir, args.image_root, transforms=val_transforms)

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    param_groups = [
        {'params': list(model.backbone.parameters()), 'lr': args.lr_backbone},
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
    logger.info(f'LR backbone: {args.lr_backbone}, LR regressor: {args.lr_regressor}')
    logger.info('Batch size: %s, Optimizer: AdamW', args.batch_size)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        logger.info(f'=== Epoch {epoch} ===')
        train_one_epoch(train_dataloader, test_dataloader, model, optimizer, epoch, device, logger)
        scheduler.step()

        ckpt_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f'Saved checkpoint: {ckpt_path}')

    logger.info('Training complete.')
    logger.info(f'Best | Range MAE: {best_range_mae:.2f} | '
                f'Heading MAE: {best_heading_mae:.2f} | Success Rate: {best_success_rate:.2f}%')


if __name__ == '__main__':
    main()
