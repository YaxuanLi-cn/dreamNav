import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import pickle
import math
import numpy as np
import torch.nn.functional as F
import logging

from torch.utils.data import DataLoader

norm_range_max = 132.0
norm_range_min = -132.0
norm_heading_max = 180.0
norm_heading_min = -180.0


def get_scheduler(optimizer, warmup_epochs, total_epochs, step_size, gamma):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs) 
        else:
            return gamma ** ((current_epoch - warmup_epochs) // step_size)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TourFrameDataset(Dataset):
    def __init__(self, json_dir, embedding_dir):
        super().__init__()
        
        self.embedding_dir = embedding_dir
        self.json_paths = [os.path.join(json_dir, g, f) for g in os.listdir(json_dir) for f in os.listdir(os.path.join(json_dir, g)) if f.endswith('.json')]

        self.images = []
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

            image_emb_a = os.path.join(self.embedding_dir, data['image_a'].replace('.jpeg', '.pkl'))
            image_emb_b = os.path.join(self.embedding_dir, data['image_b'].replace('.jpeg', '.pkl'))

            with open(image_emb_a, "rb") as f:
                image_a_emb = pickle.load(f)

            with open(image_emb_b, "rb") as f:
                image_b_emb = pickle.load(f)

            image_a_emb = image_a_emb.squeeze(0).float()
            image_b_emb = image_b_emb.squeeze(0).float()

            image_emb = torch.cat((image_a_emb, image_b_emb), dim=0)

            self.all_json_path.append(json_path)
            self.images.append(image_emb)

            theta_deg = float(data['heading_num']) 
            
            self.label_heading.append(theta_deg)
            norm_heading = (theta_deg - norm_heading_min) / (norm_heading_max - norm_heading_min)

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
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label_heading = torch.tensor(self.label_heading[idx], dtype=torch.float32) 
        label_norm_heading = torch.tensor(self.label_norm_heading[idx], dtype=torch.float32)
        label_range = torch.tensor(self.label_ranges[idx], dtype=torch.float32) 
        label_norm_range = torch.tensor(self.label_norm_ranges[idx], dtype=torch.float32) 

        json_path = self.all_json_path[idx]
        return image, label_heading, label_norm_heading, json_path, label_range, label_norm_range


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
parser.add_argument('--lr_regressor', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=1e-10, type=float, dest='weight_decay')
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--warmup_epochs', default=1, type=int)
parser.add_argument('--embedding_dir', type=str, required=True, help='Path to embedding directory')
parser.add_argument('--train_dir', type=str, required=True, help='Path to training set JSON directory')
parser.add_argument('--test_dir', type=str, required=True, help='Path to test set JSON directory')
parser.add_argument('--output_file', type=str, default='test_results.log', help='Path to output file for test results')

class OurModel(nn.Module):
    def __init__(self, pretrained=True):
        super(OurModel, self).__init__()

        self.regressor = nn.Linear(8192, 2) 

    def forward(self, images_a):
        
        output = self.regressor(images_a)

        return output 


def validate(test_loader, model, device):
    model.eval()
    all_heading_errors = []
    all_range_errors = []
    all_success = []

    with torch.no_grad():
        for i, (image_a, label_heading, label_norm_heading, _, label_range, label_norm_range) in enumerate(tqdm(test_loader, desc='[Validate]')):
            image_a = image_a.to(device, non_blocking=True)
            label_heading = label_heading.to(device, non_blocking=True)
            label_norm_heading = label_norm_heading.to(device, non_blocking=True)
            label_range = label_range.to(device, non_blocking=True)
            label_norm_range = label_norm_range.to(device, non_blocking=True)

            output = model(image_a)
            pred_range = output[:,0] * (norm_range_max - norm_range_min) + norm_range_min
            pred_heading = output[:,1] * (norm_heading_max - norm_heading_min) + norm_heading_min

            # Range MAE
            all_range_errors.append(torch.abs(pred_range - label_range))

            # Circular heading MAE: e.g. 359 vs 1 -> diff=2, not 358
            heading_diff = torch.abs(pred_heading - label_heading)
            heading_diff = heading_diff % 360.0
            heading_diff = torch.min(heading_diff, 360.0 - heading_diff)
            all_heading_errors.append(heading_diff)

            # Success rate: Euclidean distance between destinations < 10m
            true_heading_rad = label_heading * math.pi / 180.0
            pred_heading_rad = pred_heading * math.pi / 180.0
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

    for i, (image_a, label_heading, label_norm_heading, _, label_range, label_norm_range) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch} [Train]')):
        
        model.train()
        
        image_a = image_a.to(device, non_blocking=True)
        label_heading = label_heading.to(device, non_blocking=True)
        label_norm_heading = label_norm_heading.to(device, non_blocking=True)
        label_range = label_range.to(device, non_blocking=True)
        label_norm_range = label_norm_range.to(device, non_blocking=True)

        output = model(image_a)
        pred_range = output[:,0]
        pred_heading = output[:,1]

        now_heading_loss = F.smooth_l1_loss(pred_heading, label_norm_heading, beta=0.1, reduction='mean')
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

    model = OurModel()
    model.to(device)

    train_dataset = TourFrameDataset(args.train_dir, args.embedding_dir)
    test_dataset = TourFrameDataset(args.test_dir, args.embedding_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    criterion = None

    optimizer = torch.optim.SGD([
        {'params': list(model.regressor.parameters()), 'lr': args.lr_regressor}
    ], momentum=args.momentum, weight_decay=args.weight_decay)

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