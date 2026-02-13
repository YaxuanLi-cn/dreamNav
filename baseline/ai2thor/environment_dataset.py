# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import random
from PIL import Image

class DatasetEnvironment(object):
  """
  Environment that uses dataset for reinforcement learning.
  Simulates a navigation task where the agent predicts heading and range.
  """
  def __init__(self, config):
    self.scene_name = config.get('scene_name', '0839')
    self.dataset_path = config.get('dataset_path', '/root/datasets/train')
    self.dataset_root = config.get('dataset_root', '/root/dreamNav/pairUAV/tours')
    self.image_size = config.get('image_size', (84, 84))
    
    # Accept pre-assigned sample paths or scan directory
    assigned_paths = config.get('sample_paths', None)
    if assigned_paths is not None:
      self.sample_paths = list(assigned_paths)
    else:
      self.sample_paths = []
      self._scan_sample_paths()
    
    # Shuffle for randomness within epoch
    random.shuffle(self.sample_paths)
    self._sample_index = 0
    self._exhausted = False
    
    # Current state
    self.current_sample = None
    self.s_t = None  # Current state (image_a features)
    self.target = None  # Target (image_b features)
    self.true_heading = None
    self.true_range = None
    self.terminal = False
    self.reward = 0
    
    # Episode tracking
    self.episode_length = 0
    self.max_episode_length = 1  # One step per episode for supervised-like training
  
  @property
  def exhausted(self):
    return self._exhausted
  
  def _scan_sample_paths(self):
    """Scan and collect JSON file paths without loading contents."""
    scene_path = os.path.join(self.dataset_path, self.scene_name)
    
    if not os.path.exists(scene_path):
      return
    
    for file in os.listdir(scene_path):
      if file.endswith('.json'):
        self.sample_paths.append(os.path.join(scene_path, file))
  
  def _load_sample(self, json_path):
    """Load a single sample from a JSON file on demand."""
    try:
      with open(json_path, 'r') as f:
        data = json.load(f)
        if all(k in data for k in ['image_a', 'image_b', 'heading_num', 'range_num']):
          return {
            'image_a': os.path.join(self.dataset_root, data['image_a']),
            'image_b': os.path.join(self.dataset_root, data['image_b']),
            'heading': float(data['heading_num']),
            'range': float(data['range_num'])
          }
    except Exception as e:
      print(f"Error loading {json_path}: {e}")
    return None
  
  def _load_image(self, image_path):
    """Load and preprocess an image."""
    try:
      img = Image.open(image_path)
      img = img.resize(self.image_size)
      if img.mode != 'RGB':
        img = img.convert('RGB')
      img_array = np.array(img, dtype=np.float32) / 255.0
      return img_array
    except Exception as e:
      print(f"Error loading image {image_path}: {e}")
      return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)
  
  def _extract_features(self, image):
    """Extract features from image to match [2048, 4] shape."""
    h, w, c = image.shape
    flattened = image.reshape(-1)
    
    target_size = 2048 * 4
    if len(flattened) < target_size:
      features = np.zeros(target_size, dtype=np.float32)
      features[:len(flattened)] = flattened
    else:
      features = flattened[:target_size]
    
    return features.reshape(2048, 4)
  
  def _set_dummy_state(self):
    """Set dummy state when no data is available."""
    self.s_t = np.zeros((2048, 4), dtype=np.float32)
    self.target = np.zeros((2048, 4), dtype=np.float32)
    self.true_heading = 0.0
    self.true_range = 0.0
    self.terminal = False
    self.reward = 0
    self.episode_length = 0
  
  def reset(self):
    """Reset environment and load next sample sequentially."""
    if self._exhausted or len(self.sample_paths) == 0:
      self._exhausted = True
      self._set_dummy_state()
      return
    
    # Load next sample sequentially
    self.current_sample = None
    while self.current_sample is None and self._sample_index < len(self.sample_paths):
      json_path = self.sample_paths[self._sample_index]
      self._sample_index += 1
      self.current_sample = self._load_sample(json_path)
    
    if self.current_sample is None:
      self._exhausted = True
      self._set_dummy_state()
      return
    
    # Load images and extract features
    img_a = self._load_image(self.current_sample['image_a'])
    img_b = self._load_image(self.current_sample['image_b'])
    
    self.s_t = self._extract_features(img_a)
    self.target = self._extract_features(img_b)
    self.true_heading = self.current_sample['heading']
    self.true_range = self.current_sample['range']
    
    self.terminal = False
    self.reward = 0
    self.episode_length = 0
  
  def step(self, action):
    """
    Take a step in the environment.
    action: [heading, range] predicted by the agent
    """
    self.episode_length += 1
    
    # Unpack action
    pred_heading, pred_range = action[0], action[1]
    
    # Compute reward based on prediction error
    # Normalize errors by max range
    heading_error = abs(pred_heading - self.true_heading) / 180.0  # Normalize by max heading
    range_error = abs(pred_range - self.true_range) / 132.0  # Normalize by max range
    
    # Reward is negative error (closer is better)
    # Use exponential reward to emphasize accuracy
    heading_reward = np.exp(-heading_error * 2)  # Scale factor of 2
    range_reward = np.exp(-range_error * 2)
    
    self.reward = (heading_reward + range_reward) / 2.0 - 0.5  # Center around 0
    
    # Episode terminates after one step (like supervised learning)
    if self.episode_length >= self.max_episode_length:
      self.terminal = True
    
    return self.reward
  
  def update(self):
    """Update state after step (required by A3C framework)."""
    # For single-step episodes, we reset after terminal
    if self.terminal:
      self.reset()
