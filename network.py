# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Actor-Critic Network Base Class
# The policy network and value network architecture
# should be implemented in a child class of this one
class ActorCriticNetwork(object):
  def __init__(self,
               action_size,
               device="/cpu:0"):
    self._device = device
    self._action_size = action_size

  def prepare_loss(self, entropy_beta, scopes):

    # drop task id (last element) as all tasks in
    # the same scene share the same output branch
    scope_key = self._get_key(scopes[:-1])

    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi[scope_key], 1e-20, 1.0))

      # policy entropy
      self.entropy = -tf.reduce_sum(self.pi[scope_key] * log_pi, axis=1)

      # policy loss (output)
      policy_loss = - tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td + self.entropy * entropy_beta)

      # R (input for value)
      self.r = tf.placeholder("float", [None])

      # value loss (output)
      # learning rate for critic is half of actor's
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v[scope_key])

      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss

  def run_policy_and_value(self, sess, s_t, task):
    raise NotImplementedError()

  def run_policy(self, sess, s_t, task):
    raise NotImplementedError()

  def run_value(self, sess, s_t, task):
    raise NotImplementedError()

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    local_src_var_names = [self._local_var_name(x) for x in src_vars]
    local_dst_var_names = [self._local_var_name(x) for x in dst_vars]

    # keep only variables from both src and dst
    src_vars = [x for x in src_vars
      if self._local_var_name(x) in local_dst_var_names]
    dst_vars = [x for x in dst_vars
      if self._local_var_name(x) in local_src_var_names]

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "ActorCriticNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # variable (global/scene/task1/W_fc:0) --> scene/task1/W_fc:0
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_weight_variable(self, shape, name='W_fc'):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_weight_variable(self, shape, name='W_conv'):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_bias_variable(self, shape, w, h, input_channels, name='b_conv'):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_variable(self, weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

  def _get_key(self, scopes):
    return '/'.join(scopes)

# Actor-Critic Feed-Forward Network
class ActorCriticFFNetwork(ActorCriticNetwork):
  """
    Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017]
    We use tf.variable_scope() to define domains for parameter sharing
  """
  def __init__(self,
               action_size,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):
    ActorCriticNetwork.__init__(self, action_size, device)

    self.pi = dict()
    self.v = dict()

    self.W_convS1 = dict()
    self.b_convS1 = dict()

    self.W_convS2 = dict()
    self.b_convS2 = dict()

    self.W_convS3 = dict()
    self.b_convS3 = dict()

    self.W_fc1 = dict()
    self.b_fc1 = dict()

    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.W_fc3 = dict()
    self.b_fc3 = dict()

    self.W_policy = dict()
    self.b_policy = dict()

    self.W_value = dict()
    self.b_value = dict()

    with tf.device(self._device):

      # state (input)
      self.s = tf.placeholder("float", [None, 210, 160, 3])

      # target (input)
      # self.t = tf.placeholder("float", [None, 2048, 4])

      with tf.variable_scope(network_scope):
        # network key
        key = network_scope

        # Convolution network
        # Conv 1
        self.W_convS1[key], self.b_convS1[key] = self._conv_variable([8, 8, 3, 16])  # stride=4
        # self.W_convT1[key], self.b_convT1[key] = self._conv_variable([8, 8, 4, 16])  # stride=4

        self.S_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_convS1[key], 4) + self.b_convS1[key])
        # self.T_conv1 = tf.nn.relu(self._conv2d(self.t,  self.W_convT1[key], 4) + self.b_convT1[key])

        # Conv 2
        self.W_convS2[key], self.b_convS2[key] = self._conv_variable([4, 4, 16, 32])  # stride=2
        # self.W_convT2[key], self.b_convT2[key] = self._conv_variable([4, 4, 16, 32])  # stride=2

        self.S_conv2 = tf.nn.relu(self._conv2d(self.S_conv1, self.W_convS2[key], 2) + self.b_convS2[key])
        # self.T_conv2 = tf.nn.relu(self._conv2d(self.T_conv1,  self.W_convT2[key], 2) + self.b_convT2[key])

        # Conv 3
        self.W_convS3[key], self.b_convS3[key] = self._conv_variable([4, 4, 32, 64])  # stride=2
        # self.W_convT3[key], self.b_convT3[key] = self._conv_variable([4, 4, 32, 64])  # stride=2

        self.S_conv3 = tf.nn.relu(self._conv2d(self.S_conv2, self.W_convS3[key], 2) + self.b_convS3[key])

        # flatten input
        # self.s_flat = tf.reshape(self.S_conv3, [-1, 8192])
        self.s_flat = tf.reshape(self.S_conv3, [-1, 5632])

        # self.t_flat = tf.reshape(self.t, [-1, 8192])

        # shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([5632, 512])
        self.b_fc1[key] = self._fc_bias_variable([512], 5632)

        h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])
        # h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1[key]) + self.b_fc1[key])
        # h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1)
        h_fc1 = tf.concat(values=[h_s_flat], axis=1)

        # shared fusion layer
        # self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        # self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        self.W_fc2[key] = self._fc_weight_variable([512, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 512)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])

        for scene_scope in scene_scopes:
          # scene-specific key
          key = self._get_key([network_scope, scene_scope])

          with tf.variable_scope(scene_scope):

            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3[key]) + self.b_fc3[key])

            # weight for policy output layer
            self.W_policy[key] = self._fc_weight_variable([512, action_size])
            self.b_policy[key] = self._fc_bias_variable([action_size], 512)

            # policy (output)
            pi_ = tf.matmul(h_fc3, self.W_policy[key]) + self.b_policy[key]
            self.pi[key] = tf.nn.softmax(pi_)

            # weight for value output layer
            self.W_value[key] = self._fc_weight_variable([512, 1])
            self.b_value[key] = self._fc_bias_variable([1], 512)

            # value (output)
            v_ = tf.matmul(h_fc3, self.W_value[key]) + self.b_value[key]
            self.v[key] = tf.reshape(v_, [-1])

  def run_policy_and_value(self, sess, state, scopes):
    k = self._get_key(scopes[:2])
    pi_out, v_out = sess.run( [self.pi[k], self.v[k]], feed_dict = {self.s : [state]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, state, scopes):
    k = self._get_key(scopes[:2])
    pi_out = sess.run( self.pi[k], feed_dict = {self.s : [state]} )
    return pi_out[0]

  def run_value(self, sess, state, scopes):
    k = self._get_key(scopes[:2])
    v_out = sess.run( self.v[k], feed_dict = {self.s : [state]} )
    return v_out[0]

  def get_vars(self):
    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_policy, self.b_policy,
      self.W_value, self.b_value
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs

# Actor-Critic Feed-Forward Network
class ActorCriticLSTMNetwork(ActorCriticNetwork):
  """
    Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017]
    We use tf.variable_scope() to define domains for parameter sharing
  """
  def __init__(self,
               action_size,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):
    ActorCriticNetwork.__init__(self, action_size, device)

    self.pi = dict()
    self.v = dict()

    self.W_convS1 = dict()
    self.b_convS1 = dict()

    self.W_convS2 = dict()
    self.b_convS2 = dict()

    self.W_convS3 = dict()
    self.b_convS3 = dict()

    self.W_fc1 = dict()
    self.b_fc1 = dict()

    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.W_fc3 = dict()
    self.b_fc3 = dict()

    self.W_policy = dict()
    self.b_policy = dict()

    self.W_value = dict()
    self.b_value = dict()

    with tf.device(self._device):

      # state (input)
      self.s = tf.placeholder("float", [None, 210, 160, 3])

      # target (input)
      # self.t = tf.placeholder("float", [None, 2048, 4])

      with tf.variable_scope(network_scope):
        # network key
        key = network_scope

        # Convolution network
        # Conv 1
        self.W_convS1[key], self.b_convS1[key] = self._conv_variable([8, 8, 3, 16])  # stride=4
        # self.W_convT1[key], self.b_convT1[key] = self._conv_variable([8, 8, 4, 16])  # stride=4

        self.S_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_convS1[key], 4) + self.b_convS1[key])
        # self.T_conv1 = tf.nn.relu(self._conv2d(self.t,  self.W_convT1[key], 4) + self.b_convT1[key])

        # Conv 2
        self.W_convS2[key], self.b_convS2[key] = self._conv_variable([4, 4, 16, 32])  # stride=2
        # self.W_convT2[key], self.b_convT2[key] = self._conv_variable([4, 4, 16, 32])  # stride=2

        self.S_conv2 = tf.nn.relu(self._conv2d(self.S_conv1, self.W_convS2[key], 2) + self.b_convS2[key])
        # self.T_conv2 = tf.nn.relu(self._conv2d(self.T_conv1,  self.W_convT2[key], 2) + self.b_convT2[key])

        # Conv 3
        self.W_convS3[key], self.b_convS3[key] = self._conv_variable([4, 4, 32, 64])  # stride=2
        # self.W_convT3[key], self.b_convT3[key] = self._conv_variable([4, 4, 32, 64])  # stride=2

        self.S_conv3 = tf.nn.relu(self._conv2d(self.S_conv2, self.W_convS3[key], 2) + self.b_convS3[key])

        # flatten input
        # self.s_flat = tf.reshape(self.S_conv3, [-1, 8192])
        self.s_flat = tf.reshape(self.S_conv3, [-1, 5632])

        # self.t_flat = tf.reshape(self.t, [-1, 8192])

        # shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([5632, 512])
        self.b_fc1[key] = self._fc_bias_variable([512], 5632)

        h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])
        # h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1[key]) + self.b_fc1[key])
        # h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1)
        h_fc1 = tf.concat(values=[h_s_flat], axis=1)

        # shared fusion layer
        # self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        # self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        self.W_fc2[key] = self._fc_weight_variable([512, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 512)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])

        for scene_scope in scene_scopes:
          # scene-specific key
          key = self._get_key([network_scope, scene_scope])

          with tf.variable_scope(scene_scope):

            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3[key]) + self.b_fc3[key])

            # lstm
            self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            # place holder for LSTM unrolling time step size.
            self.step_size = tf.placeholder(tf.float32, [1])

            self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
            self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
            self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                                    self.initial_lstm_state1)

            # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
            # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
            # Unrolling step size is applied via self.step_size placeholder.
            # When forward propagating, step_size is 1.
            # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                              h_fc3,
                                                              initial_state=self.initial_lstm_state,
                                                              sequence_length=self.step_size,
                                                              time_major=False)

            lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])

            # weight for policy output layer
            self.W_policy[key] = self._fc_weight_variable([512, action_size])
            self.b_policy[key] = self._fc_bias_variable([action_size], 512)

            # policy (output)
            pi_ = tf.matmul(lstm_outputs, self.W_policy[key]) + self.b_policy[key]
            self.pi[key] = tf.nn.softmax(pi_)

            # weight for value output layer
            self.W_value[key] = self._fc_weight_variable([512, 1])
            self.b_value[key] = self._fc_bias_variable([1], 512)

            # value (output)
            v_ = tf.matmul(lstm_outputs, self.W_value[key]) + self.b_value[key]
            self.v[key] = tf.reshape(v_, [-1])

            self.reset_state()

  def reset_state(self):
    self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                        np.zeros([1, 256]))

  def run_policy_and_value(self, sess, state, scopes):
    k = self._get_key(scopes[:2])
    # pi_out, v_out = sess.run( [self.pi[k], self.v[k]], feed_dict = {self.s : [state]} )
    pi_out, v_out, self.lstm_state_out = sess.run([self.pi[k], self.v[k], self.lstm_state],
                                                  feed_dict={self.s: [state],
                                                             self.initial_lstm_state0: self.lstm_state_out[0],
                                                             self.initial_lstm_state1: self.lstm_state_out[1],
                                                             self.step_size: [1]})
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, state, scopes):
    k = self._get_key(scopes[:2])
    # pi_out = sess.run( self.pi[k], feed_dict = {self.s : [state]} )
    pi_out, self.lstm_state_out = sess.run([self.pi[k], self.lstm_state],
                                           feed_dict={self.s: [state],
                                                      self.initial_lstm_state0: self.lstm_state_out[0],
                                                      self.initial_lstm_state1: self.lstm_state_out[1],
                                                      self.step_size: [1]})
    return pi_out[0]

  def run_value(self, sess, state, scopes):
    k = self._get_key(scopes[:2])
    # v_out = sess.run( self.v[k], feed_dict = {self.s : [state]} )
    prev_lstm_state_out = self.lstm_state_out
    v_out, _ = sess.run([self.v[k], self.lstm_state],
                        feed_dict={self.s: [state],
                                   self.initial_lstm_state0: self.lstm_state_out[0],
                                   self.initial_lstm_state1: self.lstm_state_out[1],
                                   self.step_size: [1]})
    # roll back lstm state
    self.lstm_state_out = prev_lstm_state_out
    return v_out[0]

  def get_vars(self):
    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_policy, self.b_policy,
      self.W_value, self.b_value
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs
