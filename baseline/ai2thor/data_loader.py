# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import random

class ImagePairDataLoader(object):
  """
  Data loader for image pair dataset.
  Loads images and their corresponding heading and range labels.
  """
  def __init__(self, dataset_path, batch_size=32, image_size=(84, 84), dataset_root=None):
    self.dataset_path = dataset_path
    self.batch_size = batch_size
    self.image_size = image_size
    self.dataset_root = dataset_root if dataset_root is not None else '/root/dreamNav/pairUAV/tours'

    # Only collect JSON file paths (lazy loading)
    self.sample_paths = []
    self._scan_sample_paths()
    
    # Shuffle paths
    random.shuffle(self.sample_paths)
    
    print(f"Found {len(self.sample_paths)} samples from {dataset_path}")
  
  def _scan_sample_paths(self):
    """Scan and collect JSON file paths without loading contents."""
    if not os.path.exists(self.dataset_path):
      raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
    
    for root, dirs, files in os.walk(self.dataset_path):
      for file in files:
        if file.endswith('.json'):
          self.sample_paths.append(os.path.join(root, file))
  
  def _load_sample(self, json_path):
    """Load a single sample from a JSON file on demand."""
    try:
      with open(json_path, 'r') as f:
        data = json.load(f)
        if all(k in data for k in ['image_a', 'image_b', 'heading_num', 'range_num']):
          a_path = os.path.join(self.dataset_root, data["image_a"])
          b_path = os.path.join(self.dataset_root, data["image_b"])
          return {
            'image_a': a_path,
            'image_b': b_path,
            'heading': float(data['heading_num']),
            'range': float(data['range_num'])
          }
    except Exception as e:
      print(f"Error loading {json_path}: {e}")
    return None
  
  def _load_image(self, image_path):
    """Load and preprocess an image."""
    try:
      # Load image
      img = Image.open(image_path)
      
      # Resize
      img = img.resize(self.image_size)
      
      # Convert to RGB if needed
      if img.mode != 'RGB':
        img = img.convert('RGB')
      
      # Convert to numpy array and normalize
      img_array = np.array(img, dtype=np.float32) / 255.0
      
      return img_array
    except Exception as e:
      print(f"Error loading image {image_path}: {e}")
      # Return zero image as fallback
      return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)
  
  def _extract_features(self, image):
    """
    Extract features from image. For now, this is a placeholder.
    In the original ICRA code, they use ResNet features.
    Here we'll flatten and process the image.
    """
    # Flatten image to match expected input shape [2048, 4]
    # This is a simplified version - you may need to use actual feature extraction
    h, w, c = image.shape
    flattened = image.reshape(-1)
    
    # Pad or truncate to get [2048, 4] shape
    target_size = 2048 * 4
    if len(flattened) < target_size:
      features = np.zeros(target_size, dtype=np.float32)
      features[:len(flattened)] = flattened
    else:
      features = flattened[:target_size]
    
    return features.reshape(2048, 4)
  
  def get_batch(self):
    """Get a random batch of data with lazy loading."""
    # Randomly sample paths and load on demand
    sampled_paths = random.sample(self.sample_paths, min(self.batch_size, len(self.sample_paths)))
    batch_samples = []
    for path in sampled_paths:
      sample = self._load_sample(path)
      if sample is not None:
        batch_samples.append(sample)
    
    batch_image_a = []
    batch_image_b = []
    batch_heading = []
    batch_range = []
    
    for sample in batch_samples:
      img_a = self._load_image(sample['image_a'])
      img_b = self._load_image(sample['image_b'])
      
      # Extract features
      feat_a = self._extract_features(img_a)
      feat_b = self._extract_features(img_b)
      
      batch_image_a.append(feat_a)
      batch_image_b.append(feat_b)
      batch_heading.append(sample['heading'])
      batch_range.append(sample['range'])
    
    return {
      'image_a': np.array(batch_image_a),
      'image_b': np.array(batch_image_b),
      'heading': np.array(batch_heading),
      'range': np.array(batch_range)
    }
  
  def get_num_samples(self):
    """Return total number of samples."""
    return len(self.sample_paths)
  
  def get_num_batches(self):
    """Return number of batches per epoch."""
    return (len(self.sample_paths) + self.batch_size - 1) // self.batch_size
