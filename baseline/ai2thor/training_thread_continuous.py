# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from utils.accum_trainer import AccumTrainer
from environment_dataset import DatasetEnvironment
from network_continuous import ActorCriticContinuousNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import VERBOSE

class A3CTrainingThreadContinuous(object):
  """A3C Training Thread for continuous action space (heading and range)."""
  
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device,
               network_scope="network",
               scene_scope="scene",
               task_scope="task",
               dataset_path=None,
               dataset_root=None,
               sample_paths=None):
    
    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step
    
    self.network_scope = network_scope
    self.scene_scope = scene_scope
    self.task_scope = task_scope
    self.dataset_path = dataset_path
    self.dataset_root = dataset_root
    self.sample_paths = sample_paths
    self.scopes = [network_scope, scene_scope, task_scope]
    
    self.local_network = ActorCriticContinuousNetwork(
                           device=device,
                           network_scope=network_scope,
                           scene_scopes=[scene_scope])
    
    self.local_network.prepare_loss(ENTROPY_BETA, self.scopes)
    
    self.trainer = AccumTrainer(device)
    self.trainer.prepare_minimize(self.local_network.total_loss,
                                  self.local_network.get_vars())
    
    self.accum_gradients = self.trainer.accumulate_gradients()
    self.reset_gradients = self.trainer.reset_gradients()
    
    accum_grad_names = [self._local_var_name(x) for x in self.trainer.get_accum_grad_list()]
    global_net_vars = [x for x in global_network.get_vars() if self._get_accum_grad_name(x) in accum_grad_names]
    
    self.apply_gradients = grad_applier.apply_gradients(
      global_net_vars, self.trainer.get_accum_grad_list())
    
    self.sync = self.local_network.sync_from(global_network)
    
    self.env = None
    
    self.local_t = 0
    
    self.initial_learning_rate = initial_learning_rate
    
    self.episode_reward = 0
    self.episode_length = 0
    self.episode_max_q = -np.inf
  
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])
  
  def _get_accum_grad_name(self, var):
    return self._local_var_name(var).replace(':', '_') + '_accum_grad:0'
  
  def _anneal_learning_rate(self, global_time_step):
    time_step_to_go = max(self.max_global_time_step - global_time_step, 0.0)
    learning_rate = self.initial_learning_rate * time_step_to_go / self.max_global_time_step
    return learning_rate
  
  def sample_action(self, mean, log_std):
    """Sample action from Gaussian distribution."""
    std = np.exp(log_std)
    action = np.random.normal(mean, std)
    
    # Clip actions to valid ranges
    action[0] = np.clip(action[0], -180, 180)  # heading
    action[1] = np.clip(action[1], -132, 132)  # range
    
    return action
  
  def _record_score(self, sess, writer, summary_op, placeholders, values, global_t):
    feed_dict = {}
    for k in placeholders:
      feed_dict[placeholders[k]] = values[k]
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    writer.add_summary(summary_str, global_t)
  
  def process(self, sess, global_t, summary_writer, summary_op, summary_placeholders):
    
    if self.env is None:
      env_config = {
        'scene_name': self.scene_scope,
      }
      if self.dataset_path is not None:
        env_config['dataset_path'] = self.dataset_path
      if self.dataset_root is not None:
        env_config['dataset_root'] = self.dataset_root
      if self.sample_paths is not None:
        env_config['sample_paths'] = self.sample_paths
      self.env = DatasetEnvironment(env_config)
      self.env.reset()
    
    # Stop if all samples have been processed
    if self.env.exhausted:
      return 0
    
    states = []
    actions = []
    rewards = []
    values = []
    targets = []
    
    terminal_end = False
    
    # Reset accumulated gradients
    sess.run(self.reset_gradients)
    
    # Copy weights from shared to local
    sess.run(self.sync)
    
    start_local_t = self.local_t
    
    # t_max times loop
    for i in range(LOCAL_T_MAX):
      if self.env.exhausted:
        break
      
      mean, log_std, value_ = self.local_network.run_policy_and_value(
        sess, self.env.s_t, self.env.target, self.scopes)
      
      # Sample action from policy
      action = self.sample_action(mean, log_std)
      
      states.append(self.env.s_t)
      actions.append(action)
      values.append(value_)
      targets.append(self.env.target)
      
      
      # Process game
      self.env.step(action)
      
      # Receive game result
      reward = self.env.reward
      terminal = self.env.terminal
      
      self.episode_reward += reward
      self.episode_length += 1
      self.episode_max_q = max(self.episode_max_q, value_)
      
      # Clip reward
      rewards.append(np.clip(reward, -1, 1))
      
      self.local_t += 1
      
      # s_t1 -> s_t
      self.env.update()
      
      if terminal:
        terminal_end = True
        
        summary_values = {
          "episode_reward_input": self.episode_reward,
          "episode_length_input": float(self.episode_length),
          "episode_max_q_input": self.episode_max_q,
          "learning_rate_input": self._anneal_learning_rate(global_t)
        }
        
        self._record_score(sess, summary_writer, summary_op, summary_placeholders,
                          summary_values, global_t)
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_max_q = -np.inf
        self.env.reset()
        
        break
    
    # Skip gradient update if no data was collected
    if len(states) == 0:
      diff_local_t = self.local_t - start_local_t
      return diff_local_t
    
    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.env.s_t, self.env.target, self.scopes)
    
    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()
    targets.reverse()
    
    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []
    batch_t = []
    
    # Compute and accumulate gradients
    for (ai, ri, si, Vi, ti) in zip(actions, rewards, states, values, targets):
      R = ri + GAMMA * R
      td = R - Vi
      
      batch_si.append(si)
      batch_a.append(ai)  # Continuous action [heading, range]
      batch_td.append(td)
      batch_R.append(R)
      batch_t.append(ti)
    
    sess.run(self.accum_gradients,
             feed_dict={
               self.local_network.s: batch_si,
               self.local_network.a: batch_a,
               self.local_network.t: batch_t,
               self.local_network.td: batch_td,
               self.local_network.r: batch_R})
    
    cur_learning_rate = self._anneal_learning_rate(global_t)
    
    sess.run(self.apply_gradients,
             feed_dict={self.learning_rate_input: cur_learning_rate})
    
    # Return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
