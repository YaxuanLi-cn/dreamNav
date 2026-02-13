#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A3C Training with Continuous Action Space for Heading and Range Prediction
Modified from original A3C to support continuous actions: heading (-180 to 180) and range (-132 to 132)
"""
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import os
import datetime
import time
import math

try:
  from tqdm import tqdm
except ImportError:
  print("Installing tqdm...")
  import subprocess
  subprocess.check_call(['pip', 'install', 'tqdm'])
  from tqdm import tqdm

from network_continuous import ActorCriticContinuousNetwork
from training_thread_continuous import A3CTrainingThreadContinuous
from data_loader import ImagePairDataLoader

from utils.ops import log_uniform
from utils.rmsprop_applier import RMSPropApplier

from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU

def write_evaluation_log(log_path, global_step, eval_results, best_heading, best_range, best_combined):
  """
  Write evaluation results to log file.
  """
  timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  
  with open(log_path, 'a') as f:
    f.write(f"\n{'='*80}\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Global Step: {global_step}\n")
    f.write(f"\nTest Results (on {eval_results['num_samples']} samples):\n")
    f.write(f"  Heading MAE: {eval_results['heading_mae']:.4f} degrees\n")
    f.write(f"  Heading MSE: {eval_results['heading_mse']:.4f}\n")
    f.write(f"  Range MAE: {eval_results['range_mae']:.4f} meters\n")
    f.write(f"  Range MSE: {eval_results['range_mse']:.4f}\n")
    f.write(f"  Combined MAE: {eval_results['combined_mae']:.4f}\n")
    f.write(f"  Success Rate (< 10m): {eval_results['success_rate']:.2%}\n")
    f.write(f"\nBest Results So Far:\n")
    f.write(f"  Best Heading MAE: {best_heading:.4f} degrees\n")
    f.write(f"  Best Range MAE: {best_range:.4f} meters\n")
    f.write(f"  Best Combined MAE: {best_combined:.4f}\n")
    f.write(f"{'='*80}\n")

def compute_position_distance(true_heading_deg, true_range, pred_heading_deg, pred_range):
  """
  Compute Euclidean distance between the position reached by
  true (heading, range) and predicted (heading, range) from the same origin.
  Heading is in degrees, range is in meters.
  """
  true_h_rad = math.radians(true_heading_deg)
  pred_h_rad = math.radians(pred_heading_deg)
  
  true_x = true_range * math.cos(true_h_rad)
  true_y = true_range * math.sin(true_h_rad)
  pred_x = pred_range * math.cos(pred_h_rad)
  pred_y = pred_range * math.sin(pred_h_rad)
  
  return math.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)

def evaluate_model(sess, global_network, test_dataset_path, scene_scopes, network_scope, dataset_root=None):
  """
  Evaluate the model on test dataset.
  Returns: dict with 'heading_mae', 'heading_mse', 'range_mae', 'range_mse', 'success_rate'
  """
  try:
    # Load test data
    test_loader = ImagePairDataLoader(test_dataset_path, batch_size=32, dataset_root=dataset_root)
    
    if test_loader.get_num_samples() == 0:
      print("Warning: No test samples found")
      return None
    
    heading_errors = []
    range_errors = []
    successes = []
    
    # Evaluate on all test samples
    num_batches = min(test_loader.get_num_batches(), 50)  # Limit to 50 batches for speed
    
    for _ in tqdm(range(num_batches), desc='Evaluating', unit='batch',
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
      batch = test_loader.get_batch()
      batch_size = len(batch['heading'])
      
      for i in range(batch_size):
        state = batch['image_a'][i]
        target = batch['image_b'][i]
        true_heading = batch['heading'][i]
        true_range = batch['range'][i]
        
        # Get prediction from network
        # Use first scene scope for evaluation
        scopes = [network_scope, scene_scopes[0]]
        mean, log_std = global_network.run_policy(sess, state, target, scopes)
        
        pred_heading = mean[0]
        pred_range = mean[1]
        
        # Calculate errors
        heading_error = abs(pred_heading - true_heading)
        range_error = abs(pred_range - true_range)
        
        heading_errors.append(heading_error)
        range_errors.append(range_error)
        
        # Success rate: position distance < 10m
        pos_dist = compute_position_distance(true_heading, true_range, pred_heading, pred_range)
        successes.append(1.0 if pos_dist < 10.0 else 0.0)
    
    # Calculate MAE, MSE, Success Rate
    heading_mae = np.mean(heading_errors)
    heading_mse = np.mean(np.square(heading_errors))
    range_mae = np.mean(range_errors)
    range_mse = np.mean(np.square(range_errors))
    combined_mae = (heading_mae + range_mae) / 2.0
    success_rate = np.mean(successes)
    
    return {
      'heading_mae': heading_mae,
      'heading_mse': heading_mse,
      'range_mae': range_mae,
      'range_mse': range_mse,
      'combined_mae': combined_mae,
      'success_rate': success_rate,
      'num_samples': len(heading_errors)
    }
  
  except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    return None

if __name__ == '__main__':

  device = "/gpu:0" if USE_GPU else "/cpu:0"
  network_scope = "continuous_nav"
  
  dataset_root = '/root/dreamNav/pairUAV/tours'
  dataset_path = '/root/dreamNav/pairUAV/train'
  test_dataset_path = '/root/dreamNav/pairUAV/test'

  # Use a single shared scene scope to avoid creating one network head per scene
  scene_scopes = ["shared"]

  # Collect all JSON sample paths across all scene directories
  all_sample_paths = []
  if os.path.exists(dataset_path):
    for scene_dir_name in sorted(os.listdir(dataset_path)):
      scene_dir = os.path.join(dataset_path, scene_dir_name)
      if os.path.isdir(scene_dir):
        for f in os.listdir(scene_dir):
          if f.endswith('.json'):
            all_sample_paths.append(os.path.join(scene_dir, f))

  random.shuffle(all_sample_paths)
  total_samples = len(all_sample_paths)
  print(f"Total training samples: {total_samples}")
  
  global_t = 0
  stop_requested = False
  
  # Track best MAE for early stopping
  best_heading_mae = float('inf')
  best_range_mae = float('inf')
  best_combined_mae = float('inf')
  
  # Create evaluation log file
  eval_log_path = os.path.join(CHECKPOINT_DIR, 'evaluation_log.txt')
  if not os.path.exists(eval_log_path):
    with open(eval_log_path, 'w') as f:
      f.write("Evaluation Results Log\n")
      f.write(f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
      f.write(f"Test Dataset: {test_dataset_path}\n")

  if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

  initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                      INITIAL_ALPHA_HIGH,
                                      INITIAL_ALPHA_LOG_RATE)

  global_network = ActorCriticContinuousNetwork(device=device,
                                                 network_scope=network_scope,
                                                 scene_scopes=scene_scopes)

  branches = [("shared", "main")]
  NUM_TASKS = 1
  actual_parallel_size = PARALLEL_SIZE

  learning_rate_input = tf.placeholder("float")
  grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                decay=RMSP_ALPHA,
                                momentum=0.0,
                                epsilon=RMSP_EPSILON,
                                clip_norm=GRAD_NORM_CLIP,
                                device=device)

  # Evenly split all sample paths among PARALLEL_SIZE threads
  training_threads = []
  for i in range(actual_parallel_size):
    start = i * total_samples // actual_parallel_size
    end = (i + 1) * total_samples // actual_parallel_size
    thread_paths = all_sample_paths[start:end]
    training_thread = A3CTrainingThreadContinuous(i, global_network, initial_learning_rate,
                                                   learning_rate_input,
                                                   grad_applier, max(total_samples, 1),
                                                   device=device,
                                                   network_scope="thread-%d" % (i+1),
                                                   scene_scope="shared",
                                                   task_scope="main",
                                                   dataset_path=dataset_path,
                                                   dataset_root=dataset_root,
                                                   sample_paths=thread_paths)
    training_threads.append(training_thread)

  # Prepare session
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                          allow_soft_placement=True))

  init = tf.global_variables_initializer()
  sess.run(init)

  # Create tensorboard summaries
  summary_op = dict()
  summary_placeholders = dict()

  for i in range(actual_parallel_size):
    scene, task = branches[i % NUM_TASKS]
    key = scene + "-" + task

    # Summary for tensorboard
    episode_reward_input = tf.placeholder("float")
    episode_length_input = tf.placeholder("float")
    episode_max_q_input = tf.placeholder("float")

    scalar_summaries = [
      tf.summary.scalar(key+"/Episode Reward", episode_reward_input),
      tf.summary.scalar(key+"/Episode Length", episode_length_input),
      tf.summary.scalar(key+"/Episode Max V", episode_max_q_input)
    ]

    summary_op[key] = tf.summary.merge(scalar_summaries)
    summary_placeholders[key] = {
      "episode_reward_input": episode_reward_input,
      "episode_length_input": episode_length_input,
      "episode_max_q_input": episode_max_q_input,
      "learning_rate_input": learning_rate_input
    }

  summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

  # Init or load checkpoint with saver
  saver = tf.train.Saver(max_to_keep=10)

  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
    tokens = checkpoint.model_checkpoint_path.split("-")
    # Set global step
    global_t = int(tokens[1])
    print(">>> global step set: {}".format(global_t))
  else:
    print("Could not find old checkpoint")


  def train_function(parallel_index):
    global global_t
    training_thread = training_threads[parallel_index]

    scene, task = branches[parallel_index % NUM_TASKS]
    key = scene + "-" + task

    while not stop_requested:
      diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                              summary_op[key], summary_placeholders[key])
      if diff_global_t == 0:
        break  # This thread's samples are exhausted
      global_t += diff_global_t

  def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True

  train_threads = []
  for i in range(actual_parallel_size):
    train_threads.append(threading.Thread(target=train_function, args=(i,)))

  signal.signal(signal.SIGINT, signal_handler)

  # Start each training thread
  for t in train_threads:
    t.start()

  # Progress bar in main thread
  pbar = tqdm(total=total_samples, initial=0, 
              desc='Training', unit='sample', 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
  
  prev_t = global_t
  try:
    while any(t.is_alive() for t in train_threads) and not stop_requested:
      time.sleep(1.0)
      curr_t = global_t
      pbar.update(curr_t - prev_t)
      prev_t = curr_t
  except KeyboardInterrupt:
    stop_requested = True
  finally:
    pbar.close()

  # Wait for all threads to finish
  for t in train_threads:
    t.join()

  print('\nTraining complete. Saving checkpoint...')
  saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=global_t)
  
  # Run evaluation on test set after training
  print('\n' + '='*60)
  print('Running evaluation on test set...')
  eval_results = evaluate_model(sess, global_network, test_dataset_path, 
                               scene_scopes, network_scope, dataset_root=dataset_root)
  
  if eval_results is not None:
    heading_mae = eval_results['heading_mae']
    range_mae = eval_results['range_mae']
    combined_mae = eval_results['combined_mae']
    num_samples = eval_results['num_samples']
    
    print(f'Test Results (on {num_samples} samples):')
    print(f'  Heading MAE: {heading_mae:.4f} degrees')
    print(f'  Heading MSE: {eval_results["heading_mse"]:.4f}')
    print(f'  Range MAE: {range_mae:.4f} meters')
    print(f'  Range MSE: {eval_results["range_mse"]:.4f}')
    print(f'  Combined MAE: {combined_mae:.4f}')
    print(f'  Success Rate (< 10m): {eval_results["success_rate"]:.2%}')
    
    # Write to log file
    write_evaluation_log(eval_log_path, global_t, eval_results,
                       heading_mae, range_mae, combined_mae)
  else:
    print('Evaluation failed or no test data available')
  
  print('='*60)
  summary_writer.close()
