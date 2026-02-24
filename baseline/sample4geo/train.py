import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import math
import numpy as np
import torch.nn.functional as F
import logging
import cv2

from torch.utils.data import DataLoader
from sample4geo.model import TimmModel
from sample4geo.dataset.university import get_transforms

norm_range_max = 132.0
norm_range_min = -132.0
norm_heading_max = 180.0
norm_heading_min = -180.0


def angle_loss_cos_sin(pred_xy, target_vec, beta=0.1):

    loss = F.smooth_l1_loss(pred_xy, target_vec, beta=beta, reduction='mean')
    return loss


def get_scheduler(optimizer, warmup_epochs, total_epochs, step_size, gamma):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs) 
        else:
            return gamma ** ((current_epoch - warmup_epochs) // step_size)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TourFrameImageDataset(Dataset):
    def __init__(self, json_dir, image_root, transforms=None):
        super().__init__()
        
        self.image_root = image_root
        self.transforms = transforms
        self.json_paths = [os.path.join(json_dir, g, f) for g in os.listdir(json_dir) for f in os.listdir(os.path.join(json_dir, g)) if f.endswith('.json')]

        self.samples = []

        for json_path in tqdm(self.json_paths, desc='Loading dataset'):
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_a_path = os.path.join(self.image_root, data['image_a'])
            image_b_path = os.path.join(self.image_root, data['image_b'])

            theta_deg = float(data['heading_num']) 
            theta_rad = math.radians(theta_deg)
            norm_heading = [math.cos(theta_rad), math.sin(theta_rad)]

            range_num = float(data['range_num']) 
            norm_range = (range_num - norm_range_min) / (norm_range_max - norm_range_min)

            self.samples.append({
                'image_a_path': image_a_path,
                'image_b_path': image_b_path,
                'heading': theta_deg,
                'norm_heading': norm_heading,
                'range': range_num,
                'norm_range': norm_range,
                'json_path': json_path,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_a = cv2.imread(sample['image_a_path'])
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

        img_b = cv2.imread(sample['image_b_path'])
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img_a = self.transforms(image=img_a)['image']
            img_b = self.transforms(image=img_b)['image']

        label_heading = torch.tensor(sample['heading'], dtype=torch.float32)
        label_norm_heading = torch.tensor(sample['norm_heading'], dtype=torch.float32)
        label_range = torch.tensor(sample['range'], dtype=torch.float32)
        label_norm_range = torch.tensor(sample['norm_range'], dtype=torch.float32)

        return img_a, img_b, label_heading, label_norm_heading, sample['json_path'], label_range, label_norm_range


json_num = []

min_test_mse = 1000000
norm_range_max = 132.0
norm_range_min = -132.0
min_range_mae = 10000000
min_heading_mae = 1000000
best_success_rate = 0.0

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--lr_regressor', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=1e-10, type=float, dest='weight_decay')
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--warmup_epochs', default=1, type=int)
parser.add_argument('--image_root', type=str, required=True, help='Path to image root directory (tours)')
parser.add_argument('--train_dir', type=str, required=True, help='Path to training set JSON directory')
parser.add_argument('--test_dir', type=str, required=True, help='Path to test set JSON directory')
parser.add_argument('--output_file', type=str, default='test_results_e2e.log', help='Path to output file for test results')
parser.add_argument('--model_name', type=str, default='convnext_base.fb_in22k_ft_in1k_384')
parser.add_argument('--img_size', type=int, default=384)
parser.add_argument('--checkpoint', type=str, default='pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth',
                    help='Path to pretrained backbone checkpoint (empty string to skip)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights (only train regressor)')


class EndToEndModel(nn.Module):
    def __init__(self, model_name, pretrained=True, img_size=384):
        super(EndToEndModel, self).__init__()

        self.backbone = TimmModel(model_name, pretrained=pretrained, img_size=img_size)
        # convnext_base outputs 1024-dim features; two images concatenated = 2048
        self.regressor = nn.Linear(2048, 3) 

    def forward(self, img_a, img_b):
        feat_a = self.backbone(img_a)
        feat_b = self.backbone(img_b)
        feat = torch.cat((feat_a, feat_b), dim=1)
        output = self.regressor(feat)
        return output


def validate(test_loader, model, device):
    model.eval()
    all_heading_errors = []
    all_range_errors = []
    all_success = []

    with torch.no_grad():
        for i, (img_a, img_b, label_heading, label_norm_heading, _, label_range, label_norm_range) in enumerate(tqdm(test_loader, desc='[Validate]')):
            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)
            label_heading = label_heading.to(device, non_blocking=True)
            label_norm_heading = label_norm_heading.to(device, non_blocking=True)
            label_range = label_range.to(device, non_blocking=True)
            label_norm_range = label_norm_range.to(device, non_blocking=True)

            output = model(img_a, img_b)
            pred_range = output[:,0] * (norm_range_max - norm_range_min) + norm_range_min
            pred_heading_vec = output[:,1:]

            # Range MAE
            all_range_errors.append(torch.abs(pred_range - label_range))

            # Circular heading MAE via atan2 on cos/sin
            cos_d = (pred_heading_vec * label_norm_heading).sum(dim=-1).clamp(-1.0, 1.0)
            sin_d = pred_heading_vec[:, 0] * label_norm_heading[:, 1] - pred_heading_vec[:, 1] * label_norm_heading[:, 0]
            delta_rad = torch.atan2(sin_d, cos_d)
            heading_diff = torch.rad2deg(delta_rad).abs()
            all_heading_errors.append(heading_diff)

            # Success rate: Euclidean distance between destinations < 10m
            true_heading_rad = label_heading * math.pi / 180.0
            pred_heading_rad = torch.atan2(pred_heading_vec[:, 1], pred_heading_vec[:, 0])
            true_x = label_range * torch.cos(true_heading_rad)
            true_y = label_range * torch.sin(true_heading_rad)
            pred_x = pred_range * torch.cos(pred_heading_rad)
            pred_y = pred_range * torch.sin(pred_heading_rad)
            dist = torch.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
            all_success.append((dist < 10.0).float())

    all_heading_errors = torch.cat(all_heading_errors)
    all_range_errors = torch.cat(all_range_errors)
    all_success = torch.cat(all_success)

    heading_mae = all_heading_errors.mean()
    range_mae = all_range_errors.mean()
    success_rate = all_success.mean() * 100.0

    return range_mae, heading_mae, success_rate

def train(train_loader, test_dataloader, model, criterion, optimizer, epoch, device, args, logger):
    global min_test_mse
    global train_losses
    global val_losses
    global pred_num
    global true_num
    global min_range_mae
    global min_heading_mae
    global best_success_rate

    model.train()

    for i, (img_a, img_b, label_heading, label_norm_heading, _, label_range, label_norm_range) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch} [Train]')):
        
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        label_heading = label_heading.to(device, non_blocking=True)
        label_norm_heading = label_norm_heading.to(device, non_blocking=True)
        label_range = label_range.to(device, non_blocking=True)
        label_norm_range = label_norm_range.to(device, non_blocking=True)

        output = model(img_a, img_b)
        pred_range = output[:,0]
        pred_heading = output[:,1:]

        now_heading_loss = angle_loss_cos_sin(pred_heading, label_norm_heading)
        now_range_loss = F.smooth_l1_loss(pred_range, label_norm_range, beta=0.1, reduction='mean')
        
        loss = now_heading_loss + now_range_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    range_mae, heading_mae, success_rate = validate(test_dataloader, model, device)

    msg = 'Epoch {:d} | Range MAE: {:.2f} | Heading MAE: {:.2f} | Success Rate: {:.2f}%'.format(
        epoch, range_mae.item(), heading_mae.item(), success_rate.item())
    logger.info(msg)

    if range_mae + heading_mae < min_test_mse:
        min_test_mse = range_mae + heading_mae
        min_range_mae = range_mae
        min_heading_mae = heading_mae
        best_success_rate = success_rate

    best_msg = 'Best so far | Range MAE: {:.2f} | Heading MAE: {:.2f} | Success Rate: {:.2f}%'.format(
        min_range_mae.item(), min_heading_mae.item(), best_success_rate.item())
    logger.info(best_msg)

    return min_test_mse


def main():

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logger for dual output (console + file)
    logger = logging.getLogger('sample4geo_e2e')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(args.output_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Build end-to-end model
    model = EndToEndModel(model_name=args.model_name, pretrained=True, img_size=args.img_size)

    # Load pretrained backbone checkpoint if provided
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info(f'Loading backbone checkpoint: {args.checkpoint}')
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.backbone.load_state_dict(state_dict, strict=False)
    
    if args.freeze_backbone:
        logger.info('Freezing backbone parameters')
        for param in model.backbone.parameters():
            param.requires_grad = False

    model.to(device)

    # Image transforms
    img_size = (args.img_size, args.img_size)
    data_config = model.backbone.get_config()
    mean = data_config["mean"]
    std = data_config["std"]
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    train_dataset = TourFrameImageDataset(args.train_dir, args.image_root, transforms=train_drone_transforms)
    test_dataset = TourFrameImageDataset(args.test_dir, args.image_root, transforms=val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = None

    # Different learning rates for backbone and regressor
    param_groups = [
        {'params': list(model.regressor.parameters()), 'lr': args.lr_regressor}
    ]
    if not args.freeze_backbone:
        param_groups.append(
            {'params': list(model.backbone.parameters()), 'lr': args.lr_backbone}
        )

    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        step_size=30,
        gamma=0.1
    )

    for epoch in range(args.epochs):

        logger.info('Epoch: {}'.format(epoch))
        train(train_dataloader, test_dataloader, model, criterion, optimizer, epoch, device, args, logger)

        scheduler.step()

if __name__ == '__main__':
    main()
