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
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from peft import LoraConfig, get_peft_model

from torch.utils.data import DataLoader

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


class TourFrameDataset(Dataset):
    def __init__(self, json_dir, tours_dir, processor):
        super().__init__()
        
        self.tours_dir = tours_dir
        self.processor = processor
        self.json_paths = [os.path.join(json_dir, g, f) for g in os.listdir(json_dir) for f in os.listdir(os.path.join(json_dir, g)) if f.endswith('.json')]

        self.image_a_paths = []
        self.image_b_paths = []
        self.label_heading = [] 
        self.label_norm_heading = []
        self.label_ranges = []
        self.label_norm_ranges = []
        self.all_json_path = []

        bug_num = 0

        for json_path in tqdm(self.json_paths):
            bug_num += 1
            #if bug_num >= 100:
            #    break

            with open(json_path, 'r') as f:
                data = json.load(f)

            image_a_path = os.path.join(self.tours_dir, data['image_a'])
            image_b_path = os.path.join(self.tours_dir, data['image_b'])

            if not os.path.exists(image_a_path) or not os.path.exists(image_b_path):
                continue

            self.all_json_path.append(json_path)
            self.image_a_paths.append(image_a_path)
            self.image_b_paths.append(image_b_path)

            theta_deg = float(data['heading_num']) 
            theta_rad = math.radians(theta_deg)
            self.label_heading.append(theta_deg)
            norm_heading = [math.cos(theta_rad), math.sin(theta_rad)]

            self.label_norm_heading.append(norm_heading)
            
            range_num = float(data['range_num']) 
            norm_range = (range_num - norm_range_min) / (norm_range_max - norm_range_min)

            self.label_norm_ranges.append(norm_range)
            self.label_ranges.append(range_num)

        self.label_heading = np.array(self.label_heading, dtype=np.float32)
        self.label_norm_heading = np.array(self.label_norm_heading, dtype=np.float32)
        self.label_ranges = np.array(self.label_ranges, dtype=np.float32)
        self.label_norm_ranges = np.array(self.label_norm_ranges, dtype=np.float32)

    def __len__(self):
        return len(self.image_a_paths)

    def __getitem__(self, idx):
        image_a = Image.open(self.image_a_paths[idx]).convert("RGB")
        image_b = Image.open(self.image_b_paths[idx]).convert("RGB")

        inputs_a = self.processor(images=image_a, return_tensors="pt")
        inputs_b = self.processor(images=image_b, return_tensors="pt")

        pixel_values_a = inputs_a["pixel_values"].squeeze(0)
        pixel_values_b = inputs_b["pixel_values"].squeeze(0)

        label_heading = torch.tensor(self.label_heading[idx], dtype=torch.float32) 
        label_norm_heading = torch.tensor(self.label_norm_heading[idx], dtype=torch.float32)
        label_range = torch.tensor(self.label_ranges[idx], dtype=torch.float32) 
        label_norm_range = torch.tensor(self.label_norm_ranges[idx], dtype=torch.float32) 

        json_path = self.all_json_path[idx]
        return pixel_values_a, pixel_values_b, label_heading, label_norm_heading, json_path, label_range, label_norm_range


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
parser.add_argument('--lr_lora', default=1e-4, type=float)
parser.add_argument('--lr_regressor', default=1e-3, type=float)
parser.add_argument('--wd', default=1e-4, type=float, dest='weight_decay')
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--warmup_epochs', default=1, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lora_r', default=16, type=int, help='LoRA rank')
parser.add_argument('--lora_alpha', default=32, type=int, help='LoRA alpha')
parser.add_argument('--lora_dropout', default=0.05, type=float, help='LoRA dropout')
parser.add_argument('--tours_dir', type=str, required=True, help='Path to tours image directory')
parser.add_argument('--model_name', type=str, default='/root/dreamNav/models/dinov3_7b', help='Path to pretrained DINOv3 model')
parser.add_argument('--train_dir', type=str, required=True, help='Path to training set JSON directory')
parser.add_argument('--test_dir', type=str, required=True, help='Path to test set JSON directory')
parser.add_argument('--output_file', type=str, default='test_results.log', help='Path to output file for test results')

class OurModel(nn.Module):
    def __init__(self, model_name, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super(OurModel, self).__init__()

        backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        backbone.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_config)
        self.backbone.print_trainable_parameters()

        hidden_size = backbone.config.hidden_size
        self.regressor = nn.Linear(hidden_size * 2, 3) 

    def forward(self, pixel_values_a, pixel_values_b):
        
        out_a = self.backbone(pixel_values=pixel_values_a).pooler_output
        out_b = self.backbone(pixel_values=pixel_values_b).pooler_output
        combined = torch.cat((out_a, out_b), dim=-1).float()
        output = self.regressor(combined)

        return output 


def validate(test_loader, model, device):
    model.eval()
    all_heading_errors = []
    all_range_errors = []
    all_success = []

    with torch.no_grad():
        for i, (pixel_values_a, pixel_values_b, label_heading, label_norm_heading, _, label_range, label_norm_range) in enumerate(tqdm(test_loader, desc='[Validate]')):
            pixel_values_a = pixel_values_a.to(device, non_blocking=True)
            pixel_values_b = pixel_values_b.to(device, non_blocking=True)
            label_heading = label_heading.to(device, non_blocking=True)
            label_norm_heading = label_norm_heading.to(device, non_blocking=True)
            label_range = label_range.to(device, non_blocking=True)
            label_norm_range = label_norm_range.to(device, non_blocking=True)

            output = model(pixel_values_a, pixel_values_b)
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

    for i, (pixel_values_a, pixel_values_b, label_heading, label_norm_heading, _, label_range, label_norm_range) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch} [Train]')):
        
        model.train()
        
        pixel_values_a = pixel_values_a.to(device, non_blocking=True)
        pixel_values_b = pixel_values_b.to(device, non_blocking=True)
        label_heading = label_heading.to(device, non_blocking=True)
        label_norm_heading = label_norm_heading.to(device, non_blocking=True)
        label_range = label_range.to(device, non_blocking=True)
        label_norm_range = label_norm_range.to(device, non_blocking=True)

        output = model(pixel_values_a, pixel_values_b)
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
    logger = logging.getLogger('dinov3')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(args.output_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    processor = AutoImageProcessor.from_pretrained(args.model_name)

    model = OurModel(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.to(device)

    train_dataset = TourFrameDataset(args.train_dir, args.tours_dir, processor)
    test_dataset = TourFrameDataset(args.test_dir, args.tours_dir, processor)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = None

    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': args.lr_lora},
        {'params': list(model.regressor.parameters()), 'lr': args.lr_regressor}
    ], weight_decay=args.weight_decay)

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