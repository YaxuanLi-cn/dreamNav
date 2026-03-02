"""
Dataset for new_step2 residual refinement.

Two modes:
  1. With step1 predictions (test / optionally train):
     - Loads step1_xxx.json with per-sample (pred_heading, pred_range).
  2. Without step1 predictions (train, default):
     - Loads GT labels and adds Gaussian noise to simulate step1 error.
     - Noise std matches empirical step1 error distribution (~40° heading, ~30 m range).
"""

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Step2Dataset(Dataset):
    """Dataset that provides (I_c, I_t, Δ0, Δ_gt) for residual refinement."""

    def __init__(
        self,
        data_dir: str,
        pair_dir: str,
        img_size: int = 224,
        step1_json: str = '',
        heading_noise_std: float = 40.0,
        range_noise_std: float = 30.0,
        is_train: bool = True,
    ):
        """
        Args:
            data_dir:    root data dir (contains tours/).
            pair_dir:    directory with building_id/item.json pair files.
            img_size:    resize target.
            step1_json:  path to step1 prediction JSON. If empty, simulate.
            heading_noise_std: std (degrees) for heading noise simulation.
            range_noise_std:   std (metres) for range noise simulation.
            is_train:    training mode (enables noise simulation & augmentation).
        """
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train
        self.heading_noise_std = heading_noise_std
        self.range_noise_std = range_noise_std

        # ImageNet normalisation constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # ── Load data ─────────────────────────────────────────────────────
        if step1_json and os.path.isfile(step1_json):
            self._load_from_step1_json(step1_json)
        else:
            self._load_from_pair_dir(pair_dir)

    def _load_from_step1_json(self, json_path: str):
        """Load from step1 prediction JSON (has both pred and GT)."""
        print(f'[Dataset] Loading step1 predictions from {json_path} ...')
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.samples = []
        n = len(data['json_path'])
        for i in tqdm(range(n), desc='Loading step1 JSON'):
            pair_json_path = data['json_path'][i]
            # load pair JSON to get image paths
            if not os.path.isfile(pair_json_path):
                continue
            with open(pair_json_path, 'r') as f:
                pair = json.load(f)

            self.samples.append({
                'image_a': os.path.join(self.data_dir, 'tours', pair['image_a']),
                'image_b': os.path.join(self.data_dir, 'tours', pair['image_b']),
                'gt_heading_deg': float(pair['heading_num']),
                'gt_range_m': float(pair['range_num']),
                'pred_heading_deg': float(data['pred_deg_num'][i]),
                'pred_range_m': float(data['pred_rag_num'][i]),
                'has_step1_pred': True,
            })
        print(f'[Dataset] Loaded {len(self.samples)} samples with step1 preds.')

    def _load_from_pair_dir(self, pair_dir: str):
        """Load from pair directory (no step1 predictions, will simulate)."""
        print(f'[Dataset] Loading pairs from {pair_dir} (will simulate Δ0) ...')
        self.samples = []
        for building_id in tqdm(sorted(os.listdir(pair_dir)), desc='Loading pairs'):
            bdir = os.path.join(pair_dir, building_id)
            if not os.path.isdir(bdir):
                continue
            for fname in sorted(os.listdir(bdir)):
                if not fname.endswith('.json'):
                    continue
                jpath = os.path.join(bdir, fname)
                with open(jpath, 'r') as f:
                    pair = json.load(f)
                self.samples.append({
                    'image_a': os.path.join(self.data_dir, 'tours', pair['image_a']),
                    'image_b': os.path.join(self.data_dir, 'tours', pair['image_b']),
                    'gt_heading_deg': float(pair['heading_num']),
                    'gt_range_m': float(pair['range_num']),
                    'has_step1_pred': False,
                })
        print(f'[Dataset] Loaded {len(self.samples)} pairs (noise-simulated Δ0).')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # ── load & preprocess images ──────────────────────────────────────
        img_c = self._load_image(s['image_a'])
        img_t = self._load_image(s['image_b'])

        # ── ground truth ──────────────────────────────────────────────────
        gt_heading_deg = s['gt_heading_deg']
        gt_range_m = s['gt_range_m']

        # ── step1 prediction (Δ0) ────────────────────────────────────────
        if s['has_step1_pred']:
            pred_heading_deg = s['pred_heading_deg']
            pred_range_m = s['pred_range_m']
        else:
            # simulate step1 error with Gaussian noise
            pred_heading_deg = gt_heading_deg + np.random.randn() * self.heading_noise_std
            pred_range_m = gt_range_m + np.random.randn() * self.range_noise_std

        return {
            'img_c': img_c,
            'img_t': img_t,
            'pred_heading_deg': torch.tensor(pred_heading_deg, dtype=torch.float32),
            'pred_range_m': torch.tensor(pred_range_m, dtype=torch.float32),
            'gt_heading_deg': torch.tensor(gt_heading_deg, dtype=torch.float32),
            'gt_range_m': torch.tensor(gt_range_m, dtype=torch.float32),
        }

    def _load_image(self, path: str) -> torch.Tensor:
        """Load image, resize, normalise, return [3, H, W] tensor."""
        img = cv2.imread(path)
        if img is None:
            # fallback: black image
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
        return img
