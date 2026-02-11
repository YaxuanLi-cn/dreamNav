import argparse
import json
import math
import os
import time
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 用于监控模型性能的工具函数
def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_gpu_memory_usage(device):
    """获取 GPU 显存使用情况（MB）"""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        return allocated, reserved
    return 0, 0

def print_model_info(model, device):
    """打印模型信息"""
    total_params, trainable_params = count_parameters(model)
    allocated, reserved = get_gpu_memory_usage(device)
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"GPU Memory Allocated: {allocated:.2f} MB")
    print(f"GPU Memory Reserved: {reserved:.2f} MB")
    print("="*60 + "\n")

def range_loss(pred_range, target_range, beta=0.1):
    loss = F.smooth_l1_loss(pred_range, target_range, beta=beta, reduction='mean')
    return loss

def angle_loss_normalized(pred_angle, target_angle, beta=0.1):
    """消融实验：直接用归一化角度计算损失，不使用 cos/sin 表示"""
    loss = F.smooth_l1_loss(pred_angle, target_angle, beta=beta, reduction='mean')
    return loss

def wrapped_range_error(pred_range, target_range):

    pred_range = pred_range.float()
    target_range = target_range.float()

    #print(pred_range, target_range)
    diff = pred_range - target_range

    mse = (diff ** 2).mean().item() 
    mae = diff.abs().mean().item()   
    return mae, mse

def wrapped_angle_error_deg_normalized(pred_norm, target_norm):
    """消融实验：从归一化角度还原为角度，然后计算误差"""
    pred_deg = pred_norm * 180.0
    true_deg = target_norm * 180.0

    delta_deg = pred_deg - true_deg
    # 将角度差映射到 [-180, 180]
    delta_deg = (delta_deg + 180.0) % 360.0 - 180.0

    mae_deg = delta_deg.abs().mean()
    mse_deg = (delta_deg ** 2).mean()

    return mae_deg, mse_deg, pred_deg, true_deg


min_test_mae_seen = 1000000
min_test_mae_unseen = 1000000
norm_range_max = 132.0
norm_range_min = -132.0

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=2021, type=int)
parser.add_argument('--seed', default=32, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float)
parser.add_argument('--lr_regressor', default=5e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=1e-10, type=float, dest='weight_decay')
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--warmup_epochs', default=1, type=int)
parser.add_argument('--model_path', default='/root/dreamNav/models/dino_resnet', type=str, help='path to dino_resnet pretrained model')
parser.add_argument('--train_path', default='/root/autodl-tmp/dreamnav/train/', type=str, help='path to train dataset dir')
parser.add_argument('--test_path', default='/root/autodl-tmp/dreamnav/test/', type=str, help='path to test dataset dir')
parser.add_argument('--data_dir', default='/root/autodl-tmp/dreamnav', type=str, help='path to dataset dir')

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoFeatureExtractor, ResNetModel
from PIL import Image
import requests

class TourFrameDataset(Dataset):
    """消融实验：不使用 matches_data，不使用 cos/sin，直接用归一化角度"""
    def __init__(self, data_dir, json_dir, model_path='/root/dreamNav/models/dino_resnet'):
        super().__init__()
        self.image_processor = AutoFeatureExtractor.from_pretrained(model_path)
        self.dataset_path = data_dir
        self.json_paths = [os.path.join(json_dir, g, f) for g in os.listdir(json_dir) for f in os.listdir(os.path.join(json_dir, g)) if f.endswith('.json')]

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, i):
        json_path = self.json_paths[i]

        # 读 JSON
        with open(json_path, "rb") as f:
            data = json.load(f)

        a_path = self.dataset_path + '/tours/' + data["image_a"]
        b_path = self.dataset_path + '/tours/' + data["image_b"]

        # 加载图像 a
        with Image.open(a_path) as im:
            im = im.convert("RGB")
            img_a = self.image_processor(images=im, return_tensors="pt").pixel_values.squeeze(0)

        # 加载图像 b
        with Image.open(b_path) as im:
            im = im.convert("RGB")
            img_b = self.image_processor(images=im, return_tensors="pt").pixel_values.squeeze(0)

        # 直接拼接两张图片：(3+3, 224, 224) = (6, 224, 224)
        final_input = torch.cat((img_a, img_b), dim=0)

        theta_deg = float(data["heading_num"])
        # 消融：直接归一化角度到 [-1, 1]，而不是用 cos/sin
        label_norm_angle = torch.tensor(theta_deg / 180.0, dtype=torch.float32)
        label_deg = torch.tensor(theta_deg, dtype=torch.float32)

        range_num = float(data['range_num'])
        norm_range = (range_num - norm_range_min) / (norm_range_max - norm_range_min)

        return final_input, label_norm_angle, label_deg, json_path, range_num, norm_range


class OurModel(nn.Module):
    def __init__(self, model_path='/root/dreamNav/models/dino_resnet', pretrained=True):
        super(OurModel, self).__init__()

        original_resnet = ResNetModel.from_pretrained(model_path)
        #for param in original_resnet.parameters():
        #    param.requires_grad = False

        # 消融：输入从 5 通道改为 6 通道（image_a 3ch + image_b 3ch）
        self.input_conv = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.model = nn.Sequential(*list(original_resnet.children())[:-1]) 
        self.regressor1 = nn.Linear(2048, 128)
        # 消融：输出 2 维（1 归一化角度 + 1 归一化距离），而非 3 维（cos, sin, range）
        self.regressor2 = nn.Linear(128, 2)  

    def forward(self, images_a):
        images_a = self.input_conv(images_a)
        hidden_states_a = self.model(images_a).last_hidden_state
        pooled_output = torch.mean(hidden_states_a, dim=[2, 3]) 
        output = self.regressor2(self.regressor1(pooled_output))
        return output 


def main():

    open("output_ablation_norm_angle.log", "w").close()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading model...', flush=True)
    model = OurModel(model_path=args.model_path)
    model.to(device)
    model = torch.nn.DataParallel(model)
    print(f'Model loaded, using {torch.cuda.device_count()} GPUs', flush=True)
    
    # 打印模型信息
    print_model_info(model, device)

    print('Loading datasets...', flush=True)
    train_dataset = TourFrameDataset(args.data_dir, args.train_path, model_path=args.model_path)
    test_dataset = TourFrameDataset(args.data_dir, args.test_path, model_path=args.model_path)
    print('Datasets loaded', flush=True)

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=2)

    criterion = None  

    optimizer = torch.optim.SGD([
        {'params': model.module.input_conv.parameters(), 'lr': args.lr_regressor},
        {'params': model.module.model.parameters(), 'lr': args.lr},
        {'params': list(model.module.regressor1.parameters()) + list(model.module.regressor2.parameters()), 'lr': args.lr_regressor}
    ], momentum=args.momentum, weight_decay=args.weight_decay)

    
    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        step_size=30,  
        gamma=0.1
    )
    
    for epoch in range(args.epochs):
        train(train_dataloader, test_dataloader, model, criterion, optimizer, epoch, device, args)
        
        scheduler.step()
    
    
def train(train_loader, test_dataloader, model, criterion, optimizer, epoch, device, args):
    
    use_accel = True
    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
    losses_dir = AverageMeter('Loss_dir', use_accel, ':.4e', Summary.NONE)  
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_dir],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    epoch_start_time = time.time()

    for i, (image_a, label_norm_angle, label_deg, _, label_range, label_norm_range) in enumerate(train_loader):
        model.train()
        data_time.update(time.time() - end)

        image_a = image_a.to(device, non_blocking=True)
        label_norm_angle = label_norm_angle.to(device, non_blocking=True)
        label_range = label_range.to(device, non_blocking=True) 
        label_norm_range = label_norm_range.to(device, non_blocking=True) 

        output = model(image_a)

        # 消融：直接用归一化角度计算损失
        angle_lossnum = angle_loss_normalized(output[:, 0], label_norm_angle)
        range_lossnum = range_loss(output[:, 1], label_norm_range) 
        loss = angle_lossnum + range_lossnum

        #print(angle_lossnum, range_lossnum)
        losses_dir.update(loss.item(), image_a.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
            
            #test_angle_mse, test_angle_mae, test_range_mse, test_range_mae, test_success_rate = validate(test_dataloader, model, criterion, args, device, quick=True, tag="seen")
            #print(f'Seen: ANGLE: mse:{test_angle_mse:.2f} | mae:{test_angle_mae:.2f} RANGE: mse: {test_range_mse:.2f} | mae: {test_range_mae:.2f} SR: {test_success_rate*100:.2f}%')
        

    # Epoch 结束后进行完整测试
    epoch_time = time.time() - epoch_start_time
    print(f'\n===== Epoch {epoch} Finished (Time: {epoch_time:.2f}s) =====')
    test_angle_mse, test_angle_mae, test_range_mse, test_range_mae, test_success_rate = validate(test_dataloader, model, criterion, args, device, quick=False, tag="seen")
    print(f'[Epoch {epoch} Summary] ANGLE: mse:{test_angle_mse:.2f} | mae:{test_angle_mae:.2f} RANGE: mse: {test_range_mse:.2f} | mae: {test_range_mae:.2f} SR: {test_success_rate*100:.2f}%')
    with open('output_ablation_norm_angle.log', 'a') as outf:
        outf.write(f'\n===== Epoch {epoch} Summary =====\n')
        outf.write(f'ANGLE: mse:{test_angle_mse:.2f} | mae:{test_angle_mae:.2f} RANGE: mse: {test_range_mse:.2f} | mae: {test_range_mae:.2f} SR: {test_success_rate*100:.2f}%\n')
        outf.write(f'=====================================\n\n')
        outf.flush()
    
    return 

def get_scheduler(optimizer, warmup_epochs, total_epochs, step_size, gamma):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)  
        else:
            return gamma ** ((current_epoch - warmup_epochs) // step_size)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def validate(val_loader, model, criterion, args, device, quick=True, tag='seen'):

    global min_test_mae_seen
    global min_test_mae_unseen

    use_accel = True
    now_deg_preds = []
    now_deg_trues = []
    now_rag_preds = []
    now_rag_trues = []

    now_jsons = []
    
    success_count = 0
    total_count = 0

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    losses_mse_deg = AverageMeter('MSE_deg', use_accel, ':.4e', Summary.NONE)
    losses_mae_deg = AverageMeter('MAE_deg', use_accel, ':.4e', Summary.NONE)
    losses_mse_rag = AverageMeter('MSE_rag', use_accel, ':.4e', Summary.NONE)
    losses_mae_rag = AverageMeter('MAE_rag', use_accel, ':.4e', Summary.NONE)
    losses_dir = AverageMeter('Loss_dir', use_accel, ':.4e', Summary.NONE)  
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_dir, losses_mse_deg, losses_mae_deg, losses_mse_rag, losses_mae_rag],
        prefix='Test: ')

    model.eval()
    total_samples = 0
    inference_start_time = time.time()
    
    with torch.no_grad():
        end = time.time()
        for i, (image_a, label_norm_angle, label_deg, json_paths, label_range, label_norm_range) in enumerate(val_loader):
            image_a = image_a.to(device, non_blocking=True)
            label_norm_angle = label_norm_angle.to(device, non_blocking=True)
            label_deg = label_deg.to(device, non_blocking=True) 
            label_range = label_range.to(device, non_blocking=True) 
            label_norm_range = label_norm_range.to(device, non_blocking=True) 

            output = model(image_a)  
            total_samples += image_a.size(0)

            # 消融：用归一化角度计算损失和误差
            loss_dir = angle_loss_normalized(output[:, 0], label_norm_angle)
            mae_deg, mse_deg, pred_deg, true_deg = wrapped_angle_error_deg_normalized(output[:, 0], label_norm_angle)

            pred_range = output[:, 1] * (norm_range_max - norm_range_min) + norm_range_min
            mae_rag, mse_rag = wrapped_range_error(pred_range, label_range)

            # 计算成功率：预测终点与真实终点距离 < 10m
            pred_rad = torch.deg2rad(pred_deg)
            true_rad = torch.deg2rad(true_deg)
            pred_x = pred_range * torch.cos(pred_rad)
            pred_y = pred_range * torch.sin(pred_rad)
            true_x = label_range * torch.cos(true_rad)
            true_y = label_range * torch.sin(true_rad)
            endpoint_dist = torch.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
            success_count += (endpoint_dist < 10.0).sum().item()
            total_count += image_a.size(0)

            losses_dir.update(loss_dir.item(), image_a.size(0))
            losses_mse_deg.update(mse_deg.item(), image_a.size(0))
            losses_mae_deg.update(mae_deg.item(), image_a.size(0))
            losses_mse_rag.update(mse_rag, image_a.size(0))
            losses_mae_rag.update(mae_rag, image_a.size(0))

            now_deg_preds.extend(pred_deg.detach().cpu().tolist())
            now_deg_trues.extend(true_deg.detach().cpu().tolist())
            now_rag_preds.extend(pred_range.detach().cpu().tolist())
            now_rag_trues.extend(label_range.detach().cpu().tolist())

            now_jsons.extend(list(json_paths))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)
                if quick:
                    break
    
    # 计算推理速度
    inference_time = time.time() - inference_start_time
    throughput = total_samples / inference_time if inference_time > 0 else 0
    print(f"Inference Throughput ({tag}): {throughput:.2f} samples/s")
    
    success_rate = success_count / total_count if total_count > 0 else 0.0
    
    if quick:
        return losses_mse_deg.avg, losses_mae_deg.avg, losses_mse_rag.avg, losses_mae_rag.avg, success_rate
    
    now_loss_all = losses_mae_deg.avg + losses_mae_rag.avg

    have_update = False
    if tag == "seen":
        if now_loss_all < min_test_mae_seen:
            have_update = True
            min_test_mae_seen = now_loss_all
    elif tag == "unseen":
        if now_loss_all < min_test_mae_unseen:
            have_update = True
            min_test_mae_unseen = now_loss_all
    
    if have_update:
        pred_deg_num = now_deg_preds
        true_deg_num = now_deg_trues
        pred_rag_num = now_rag_preds
        true_rag_num = now_rag_trues
        json_num = now_jsons

        now_json = {'pred_deg_num': pred_deg_num, 'true_deg_num': true_deg_num, 'pred_rag_num': pred_rag_num, 'true_rag_num': true_rag_num, 'json_path': json_num}
        with open('step1_ablation_norm_angle_' + tag + '.json', 'w') as f:
            json.dump(now_json, f)
        
    return losses_mse_deg.avg, losses_mae_deg.avg, losses_mse_rag.avg, losses_mae_rag.avg, success_rate

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):    
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
