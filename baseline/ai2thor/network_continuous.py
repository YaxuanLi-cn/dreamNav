# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Actor-Critic Network for Continuous Action Space
class ActorCriticContinuousNetwork(object):
  """
  Implementation of Actor-Critic network with continuous action space.
  Actions: heading (-180 to 180) and range (-132 to 132)
  """
  def __init__(self,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):
    self._device = device
    self._network_scope = network_scope
    
    self.pi_mean = dict()  # Mean of action distribution
    self.pi_log_std = dict()  # Log standard deviation of action distribution
    self.v = dict()  # Value function
    
    self.W_fc1 = dict()
    self.b_fc1 = dict()
    self.W_fc2 = dict()
    self.b_fc2 = dict()
    self.W_fc3 = dict()
    self.b_fc3 = dict()
    
    self.W_policy_mean = dict()
    self.b_policy_mean = dict()
    self.W_policy_log_std = dict()
    self.b_policy_log_std = dict()
    
    self.W_value = dict()
    self.b_value = dict()
    
    with tf.device(self._device):
      # Input placeholders
      self.s = tf.placeholder("float", [None, 2048, 4], name="state")
      self.t = tf.placeholder("float", [None, 2048, 4], name="target")
      
      with tf.variable_scope(network_scope):
        key = network_scope
        
        # Flatten input
        self.s_flat = tf.reshape(self.s, [-1, 8192])
        self.t_flat = tf.reshape(self.t, [-1, 8192])
        
        # Shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([8192, 512])
        self.b_fc1[key] = self._fc_bias_variable([512], 8192)
        
        h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1)
        
        # Shared fusion layer
        self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])
        
        for scene_scope in scene_scopes:
          key = self._get_key([network_scope, scene_scope])
          
          with tf.variable_scope(scene_scope):
            # Scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3[key]) + self.b_fc3[key])
            
            # Policy output: mean of Gaussian distribution (2D: heading, range)
            self.W_policy_mean[key] = self._fc_weight_variable([512, 2])
            self.b_policy_mean[key] = self._fc_bias_variable([2], 512)
            self.pi_mean[key] = tf.matmul(h_fc3, self.W_policy_mean[key]) + self.b_policy_mean[key]
            
            # Policy output: log std of Gaussian distribution
            self.W_policy_log_std[key] = self._fc_weight_variable([512, 2])
            self.b_policy_log_std[key] = self._fc_bias_variable([2], 512)
            self.pi_log_std[key] = tf.matmul(h_fc3, self.W_policy_log_std[key]) + self.b_policy_log_std[key]
            
            # Value output
            self.W_value[key] = self._fc_weight_variable([512, 1])
            self.b_value[key] = self._fc_bias_variable([1], 512)
            v_ = tf.matmul(h_fc3, self.W_value[key]) + self.b_value[key]
            self.v[key] = tf.reshape(v_, [-1])
  
  def prepare_loss(self, entropy_beta, scopes):
    """Prepare loss function for continuous action space."""
    scope_key = self._get_key(scopes[:-1])
    
    with tf.device(self._device):
      # Action taken (input for policy) - 2D continuous action
      self.a = tf.placeholder("float", [None, 2])
      
      # Temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])
      
      # Compute log probability of action under current policy
      # Gaussian policy: log_prob = -0.5 * ((a - mean) / std)^2 - log(std) - 0.5*log(2*pi)
      mean = self.pi_mean[scope_key]
      log_std = self.pi_log_std[scope_key]
      std = tf.exp(log_std)
      
      # Log probability
      log_prob = -0.5 * tf.reduce_sum(tf.square((self.a - mean) / (std + 1e-8)), axis=1) \
                 - tf.reduce_sum(log_std, axis=1) \
                 - np.log(2.0 * np.pi)
      
      # Entropy for exploration (higher entropy = more exploration)
      entropy = tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=1)
      
      # Policy loss (output) - negative because we want to maximize
      policy_loss = -tf.reduce_sum(log_prob * self.td + entropy * entropy_beta)
      
      # R (input for value)
      self.r = tf.placeholder("float", [None])
      
      # Value loss (output)
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v[scope_key])
      
      # Total loss
      self.total_loss = policy_loss + value_loss
  
  def run_policy_and_value(self, sess, state, target, scopes):
    """Run policy and value networks."""
    k = self._get_key(scopes[:2])
    mean, log_std, v_out = sess.run(
      [self.pi_mean[k], self.pi_log_std[k], self.v[k]],
      feed_dict={self.s: [state], self.t: [target]}
    )
    return (mean[0], log_std[0], v_out[0])
  
  def run_policy(self, sess, state, target, scopes):
    """Run policy network."""
    k = self._get_key(scopes[:2])
    mean, log_std = sess.run(
      [self.pi_mean[k], self.pi_log_std[k]],
      feed_dict={self.s: [state], self.t: [target]}
    )
    return (mean[0], log_std[0])
  
  def run_value(self, sess, state, target, scopes):
    """Run value network."""
    k = self._get_key(scopes[:2])
    v_out = sess.run(self.v[k], feed_dict={self.s: [state], self.t: [target]})
    return v_out[0]
  
  def get_vars(self):
    """Get all trainable variables."""
    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_policy_mean, self.b_policy_mean,
      self.W_policy_log_std, self.b_policy_log_std,
      self.W_value, self.b_value
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs
  
  def sync_from(self, src_network, name=None):
    """Sync weights from source network."""
    src_vars = src_network.get_vars()
    dst_vars = self.get_vars()
    
    local_src_var_names = [self._local_var_name(x) for x in src_vars]
    local_dst_var_names = [self._local_var_name(x) for x in dst_vars]
    
    src_vars = [x for x in src_vars if self._local_var_name(x) in local_dst_var_names]
    dst_vars = [x for x in dst_vars if self._local_var_name(x) in local_src_var_names]
    
    sync_ops = []
    with tf.device(self._device):
      with tf.name_scope(name, "ActorCriticNetwork", []) as name:
        for (src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)
        return tf.group(*sync_ops, name=name)
  
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])
  
  def _fc_weight_variable(self, shape, name='W_fc'):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)
  
  def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)
  
  def _get_key(self, scopes):
    return '/'.join(scopes)
