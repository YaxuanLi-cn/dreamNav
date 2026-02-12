import os
import json
import torch
import numpy as np
import cv2
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ldm.models.diffusion.ddim import DDIMSampler

# Target output size (what user wants)
TARGET_SIZE = 64
# Processing size (must be divisible by 64 for UNet compatibility: 64/8=8 latent)
PROCESS_SIZE = 64


def pad_to_size(img, target_size, pad_value=0):
    """Pad image to target_size with center alignment"""
    h, w = img.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if len(img.shape) == 3:
        return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=pad_value)
    return np.pad(img, ((top, bottom), (left, right)), mode='constant', constant_values=pad_value)


def center_crop_tensor(tensor, target_size):
    """Center crop a tensor from PROCESS_SIZE to target_size. Tensor shape: B, C, H, W"""
    _, _, h, w = tensor.shape
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    return tensor[:, :, start_h:start_h+target_size, start_w:start_w+target_size]


def angular_difference(angle1, angle2):
    """Calculate the minimum angular difference between two angles (in degrees).
    Handles wraparound, e.g., 359° and 1° have a difference of 2°, not 358°.
    """
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)


class TestDatasetWithPrediction(Dataset):
    """Test dataset that loads predictions from step1_seen.json"""
    
    def __init__(self, step1_json_path, data_root='/root/autodl-tmp/dreamnav/try_test/'):
        self.data = []
        self.data_root = data_root
        
        # Load step1 predictions
        with open(step1_json_path, 'r') as f:
            step1_data = json.load(f)
        
        pred_deg = step1_data['pred_deg_num']
        true_deg = step1_data['true_deg_num']
        pred_rag = step1_data['pred_rag_num']
        true_rag = step1_data['true_rag_num']
        json_paths = step1_data['json_path']
        
        for i, json_path in enumerate(json_paths):
            with open(json_path, 'r', encoding='utf-8') as f:
                item_json = json.load(f)
            
            a_path = self.data_root + '/tours/' + item_json["image_a"]
            b_path = self.data_root + '/tours/' + item_json["image_b"]

            self.data.append({
                'image_a': a_path,
                'image_b': b_path,
                'pred_heading': pred_deg[i],
                'true_heading': true_deg[i],
                'pred_range': pred_rag[i],
                'true_range': true_rag[i],
                'json_path': json_path
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        source = cv2.imread(item['image_a'])
        target = cv2.imread(item['image_b'])
        
        # Resize images to TARGET_SIZE x TARGET_SIZE first
        source = cv2.resize(source, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # Pad to PROCESS_SIZE for UNet compatibility (centered)
        source = pad_to_size(source, PROCESS_SIZE, pad_value=0)
        target = pad_to_size(target, PROCESS_SIZE, pad_value=0)
        
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        return {
            'hint': source,
            'jpg': target,
            'pred_heading': item['pred_heading'],
            'true_heading': item['true_heading'],
            'pred_range': item['pred_range'],
            'true_range': item['true_range'],
        }


class EpochTestCallback(Callback):
    """Callback to run testing after each epoch with instruction correction"""
    
    def __init__(self, 
                 data_root,
                 step1_json_path='/root/dreamNav/step1/step1_seen.json',
                 heading_offset=20.0,
                 range_offset=1.5,
                 ddim_steps=50,
                 batch_size=1,
                 max_test_samples=None,
                 save_dir=None,
                 test_batch_size=3):
        """
        Args:
            step1_json_path: Path to step1_seen.json with predictions
            heading_offset: The offset value for heading correction (will try ±offset and 0)
            range_offset: The offset value for range correction (will try ±offset and 0)
            ddim_steps: Number of DDIM sampling steps
            batch_size: Batch size for testing
            max_test_samples: Maximum number of test samples (None for all)
            save_dir: Directory to save test results
            test_batch_size: Number of test samples to process simultaneously (each has 9 offset combos)
        """
        super().__init__()
        self.data_root=data_root
        self.step1_json_path = step1_json_path
        self.heading_offset = heading_offset
        self.range_offset = range_offset
        self.ddim_steps = ddim_steps
        self.batch_size = batch_size
        self.max_test_samples = max_test_samples
        self.save_dir = save_dir
        self.test_dataset = None
        self.test_batch_size = test_batch_size
        self.num_offsets = 9  # 3 heading × 3 range
        
    def setup(self, trainer, pl_module, stage=None):
        """Load test dataset once"""
        if self.test_dataset is None:
            print(f"Loading test dataset from {self.step1_json_path}...")
            self.test_dataset = TestDatasetWithPrediction(self.step1_json_path, self.data_root)
            print(f"Test dataset loaded with {len(self.test_dataset)} samples")
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Run testing at the end of each epoch"""
        # Save checkpoint before testing
        ckpt_dir = os.path.join(trainer.default_root_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'epoch={trainer.current_epoch}.ckpt')
        trainer.save_checkpoint(ckpt_path)
        print(f"\nCheckpoint saved to {ckpt_path}")

        print(f"\n{'='*60}")
        print(f"Running epoch {trainer.current_epoch} testing with heading_offset={self.heading_offset}, range_offset={self.range_offset}")
        print(f"{'='*60}")
        
        pl_module.eval()
        
        # Get test samples
        test_indices = range(len(self.test_dataset))
        if self.max_test_samples is not None:
            test_indices = range(min(self.max_test_samples, len(self.test_dataset)))
        
        # Metrics storage
        original_errors_heading = []
        corrected_errors_heading = []
        original_errors_range = []
        corrected_errors_range = []
        best_offsets_heading = []
        best_offsets_range = []
        
        device = pl_module.device
        ddim_sampler = DDIMSampler(pl_module)
        
        # Precompute all 9 offset combinations (3 heading × 3 range)
        heading_offsets = [-self.heading_offset, 0, self.heading_offset]
        range_offsets = [-self.range_offset, 0, self.range_offset]
        offset_combos = [(h_off, r_off) for h_off in heading_offsets for r_off in range_offsets]
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(test_indices), self.test_batch_size), desc="Testing"):
                batch_idx_list = list(test_indices[batch_start:batch_start + self.test_batch_size])
                actual_bs = len(batch_idx_list)
                total_bs = actual_bs * self.num_offsets  # e.g. 3 samples × 9 offsets = 27
                
                # Collect data for all samples in this batch
                all_hints = []
                all_targets = []
                all_headings = []  # (total_bs,)
                all_ranges = []    # (total_bs,)
                batch_pred_headings = []
                batch_true_headings = []
                batch_pred_ranges = []
                batch_true_ranges = []
                
                for idx in batch_idx_list:
                    sample = self.test_dataset[idx]
                    
                    hint = torch.from_numpy(sample['hint']).permute(2, 0, 1).contiguous().float()  # C, H, W
                    target = torch.from_numpy(sample['jpg']).permute(2, 0, 1).contiguous().float()  # C, H, W
                    
                    pred_heading = sample['pred_heading']
                    true_heading = sample['true_heading']
                    pred_range = sample['pred_range']
                    true_range = sample['true_range']
                    
                    batch_pred_headings.append(pred_heading)
                    batch_true_headings.append(true_heading)
                    batch_pred_ranges.append(pred_range)
                    batch_true_ranges.append(true_range)
                    
                    # Repeat hint and target 9 times (one per offset combo)
                    for h_off, r_off in offset_combos:
                        all_hints.append(hint)
                        all_targets.append(target)
                        all_headings.append(pred_heading + h_off)
                        all_ranges.append(pred_range + r_off)
                
                # Stack into batched tensors
                hints_batch = torch.stack(all_hints).to(device)       # (total_bs, C, H, W)
                targets_batch = torch.stack(all_targets).to(device)   # (total_bs, C, H, W)
                heading_tensor = torch.tensor(all_headings, device=device).float()  # (total_bs,)
                range_tensor = torch.tensor(all_ranges, device=device).float()      # (total_bs,)
                
                # Encode numeric conditions for all items at once
                cond_emb = pl_module.numeric_encoder(heading_tensor, range_tensor)  # (total_bs, seq_len, dim)
                
                # Prepare batched conditioning
                cond = {
                    "c_concat": [hints_batch],
                    "c_crossattn": [cond_emb]
                }
                
                # Unconditional conditioning for CFG
                uc_cross = pl_module.get_unconditional_conditioning(total_bs)
                uc_full = {
                    "c_concat": [hints_batch],
                    "c_crossattn": [uc_cross]
                }
                
                # Single batched DDIM sampling call
                shape = (pl_module.channels, hints_batch.shape[2] // 8, hints_batch.shape[3] // 8)
                samples, _ = ddim_sampler.sample(
                    self.ddim_steps, total_bs, shape, cond,
                    verbose=False,
                    unconditional_guidance_scale=9.0,
                    unconditional_conditioning=uc_full
                )
                
                # Decode all at once
                generated = pl_module.decode_first_stage(samples)  # (total_bs, C, H, W)
                
                # Center crop for fair comparison
                generated_cropped = center_crop_tensor(generated, TARGET_SIZE)
                targets_cropped = center_crop_tensor(targets_batch, TARGET_SIZE)
                
                # Compute per-item MSE: (total_bs,)
                mse_per_item = torch.mean((generated_cropped - targets_cropped) ** 2, dim=(1, 2, 3))
                
                # Reshape to (actual_bs, 9) to find best offset per sample
                mse_per_sample = mse_per_item.view(actual_bs, self.num_offsets)
                best_indices = mse_per_sample.argmin(dim=1)  # (actual_bs,)
                
                # Extract results per sample
                for i in range(actual_bs):
                    best_idx = best_indices[i].item()
                    best_h_off, best_r_off = offset_combos[best_idx]
                    
                    pred_heading = batch_pred_headings[i]
                    true_heading = batch_true_headings[i]
                    pred_range = batch_pred_ranges[i]
                    true_range = batch_true_ranges[i]
                    
                    best_heading = pred_heading + best_h_off
                    best_range = pred_range + best_r_off
                    
                    # Calculate errors
                    original_heading_error = angular_difference(pred_heading, true_heading)
                    corrected_heading_error = angular_difference(best_heading, true_heading)
                    original_range_error = abs(pred_range - true_range)
                    corrected_range_error = abs(best_range - true_range)
                    
                    original_errors_heading.append(original_heading_error)
                    corrected_errors_heading.append(corrected_heading_error)
                    original_errors_range.append(original_range_error)
                    corrected_errors_range.append(corrected_range_error)
                    best_offsets_heading.append(best_h_off)
                    best_offsets_range.append(best_r_off)
        
        # Calculate metrics
        original_mae_heading = np.mean(original_errors_heading)
        corrected_mae_heading = np.mean(corrected_errors_heading)
        original_mse_heading = np.mean(np.array(original_errors_heading) ** 2)
        corrected_mse_heading = np.mean(np.array(corrected_errors_heading) ** 2)
        
        original_mae_range = np.mean(original_errors_range)
        corrected_mae_range = np.mean(corrected_errors_range)
        original_mse_range = np.mean(np.array(original_errors_range) ** 2)
        corrected_mse_range = np.mean(np.array(corrected_errors_range) ** 2)
        
        # Count offset usage
        offset_counts_heading = {o: best_offsets_heading.count(o) for o in heading_offsets}
        offset_counts_range = {o: best_offsets_range.count(o) for o in range_offsets}
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Epoch {trainer.current_epoch} Test Results (heading_offset={self.heading_offset}, range_offset={self.range_offset})")
        print(f"{'='*60}")
        print(f"\nHeading:")
        print(f"  Original MAE: {original_mae_heading:.4f}")
        print(f"  Corrected MAE: {corrected_mae_heading:.4f}")
        print(f"  Original MSE: {original_mse_heading:.4f}")
        print(f"  Corrected MSE: {corrected_mse_heading:.4f}")
        print(f"  Improvement MAE: {original_mae_heading - corrected_mae_heading:.4f}")
        print(f"  Offset usage: {offset_counts_heading}")
        
        print(f"\nRange:")
        print(f"  Original MAE: {original_mae_range:.4f}")
        print(f"  Corrected MAE: {corrected_mae_range:.4f}")
        print(f"  Original MSE: {original_mse_range:.4f}")
        print(f"  Corrected MSE: {corrected_mse_range:.4f}")
        print(f"  Improvement MAE: {original_mae_range - corrected_mae_range:.4f}")
        print(f"  Offset usage: {offset_counts_range}")
        print(f"{'='*60}\n")
        
        # Log to trainer
        pl_module.log('test/original_mae_heading', original_mae_heading, on_epoch=True)
        pl_module.log('test/corrected_mae_heading', corrected_mae_heading, on_epoch=True)
        pl_module.log('test/original_mse_heading', original_mse_heading, on_epoch=True)
        pl_module.log('test/corrected_mse_heading', corrected_mse_heading, on_epoch=True)
        pl_module.log('test/original_mae_range', original_mae_range, on_epoch=True)
        pl_module.log('test/corrected_mae_range', corrected_mae_range, on_epoch=True)
        pl_module.log('test/original_mse_range', original_mse_range, on_epoch=True)
        pl_module.log('test/corrected_mse_range', corrected_mse_range, on_epoch=True)
        
        # Save results to file
        if self.save_dir:
            results = {
                'epoch': trainer.current_epoch,
                'heading_offset': self.heading_offset,
                'range_offset': self.range_offset,
                'heading': {
                    'original_mae': original_mae_heading,
                    'corrected_mae': corrected_mae_heading,
                    'original_mse': original_mse_heading,
                    'corrected_mse': corrected_mse_heading,
                    'offset_counts': offset_counts_heading
                },
                'range': {
                    'original_mae': original_mae_range,
                    'corrected_mae': corrected_mae_range,
                    'original_mse': original_mse_range,
                    'corrected_mse': corrected_mse_range,
                    'offset_counts': offset_counts_range
                }
            }
            os.makedirs(self.save_dir, exist_ok=True)
            result_path = os.path.join(self.save_dir, f'test_results_epoch_{trainer.current_epoch}.json')
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {result_path}")
        
        pl_module.train()
