# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images 图片. [batch_size, image_size, image_size, 3]
      labels: Batches of labels 类别标签. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  # 构建模型图
  def build_graph(self):
    # 新建全局step
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    # 构建ResNet网络模型
    self._build_model()
    # 构建优化训练操作
    if self.mode == 'train':
      self._build_train_op()
    # 合并所有总结
    self.summaries = tf.summary.merge_all()


  # 构建模型
  def _build_model(self):
    with tf.variable_scope('init'):
      x = self._images
      """第一层卷积（3,3x3/1,16）"""
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

    # 残差网络参数
    strides = [1, 2, 2]
    # 激活前置
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
      # bottleneck残差单元模块
      res_func = self._bottleneck_residual
      # 通道数量
      filters = [16, 64, 128, 256]
    else:
      # 标准残差单元模块
      res_func = self._residual
      # 通道数量
      filters = [16, 16, 32, 64]

    # 第一组
    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], 
                   self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    # 第二组
    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], 
                   self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)
        
    # 第三组
    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    # 全局池化层
    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    # 全连接层 + Softmax
    with tf.variable_scope('logit'):
      logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    # 构建损失函数
    with tf.variable_scope('costs'):
      # 交叉熵
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=self.labels)
      # 加和
      self.cost = tf.reduce_mean(xent, name='xent')
      # L2正则，权重衰减
      self.cost += self._decay()
      # 添加cost总结，用于Tensorborad显示
      tf.summary.scalar('cost', self.cost)

  # 构建训练操作
  def _build_train_op(self):
    # 学习率/步长
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.summary.scalar('learning_rate', self.lrn_rate)

    # 计算训练参数的梯度
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    # 设置优化方法
    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

    # 梯度优化操作
    apply_op = optimizer.apply_gradients(
                        zip(grads, trainable_variables),
                        global_step=self.global_step, 
                        name='train_step')
    
    # 合并BN更新操作
    train_ops = [apply_op] + self._extra_train_ops
    # 建立优化操作组
    self.train_op = tf.group(*train_ops)


  # 把步长值转换成tf.nn.conv2d需要的步长数组
  def _stride_arr(self, stride):    
    return [1, stride, stride, 1]

  # 残差单元模块
  def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
    # 是否前置激活(取残差直连之前进行BN和ReLU）
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        # 先做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        # 获取残差直连
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        # 获取残差直连
        orig_x = x
        # 后做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    # 第1子层
    with tf.variable_scope('sub1'):
      # 3x3卷积，使用输入步长，通道数(in_filter -> out_filter)
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    # 第2子层
    with tf.variable_scope('sub2'):
      # BN和ReLU激活
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      # 3x3卷积，步长为1，通道数不变(out_filter)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
    
    # 合并残差层
    with tf.variable_scope('sub_add'):
      # 当通道数有变化时
      if in_filter != out_filter:
        # 均值池化，无补零
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        # 通道补零(第4维前后对称补零)
        orig_x = tf.pad(orig_x, 
                        [[0, 0], 
                         [0, 0], 
                         [0, 0],
                         [(out_filter-in_filter)//2, (out_filter-in_filter)//2]
                        ])
      # 合并残差
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  # bottleneck残差单元模块
  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    # 是否前置激活(取残差直连之前进行BN和ReLU）
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        # 先做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        # 获取残差直连
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        # 获取残差直连
        orig_x = x
        # 后做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    # 第1子层
    with tf.variable_scope('sub1'):
      # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter/4)
      x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

    # 第2子层
    with tf.variable_scope('sub2'):
      # BN和ReLU激活
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      # 3x3卷积，步长为1，通道数不变(out_filter/4)
      x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    # 第3子层
    with tf.variable_scope('sub3'):
      # BN和ReLU激活
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      # 1x1卷积，步长为1，通道数不变(out_filter/4 -> out_filter)
      x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    # 合并残差层
    with tf.variable_scope('sub_add'):
      # 当通道数有变化时
      if in_filter != out_filter:
        # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter)
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      
      # 合并残差
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x


  # Batch Normalization批归一化
  # ((x-mean)/var)*gamma+beta
  def _batch_norm(self, name, x):
    with tf.variable_scope(name):
      # 输入通道维数
      params_shape = [x.get_shape()[-1]]
      # offset
      beta = tf.get_variable('beta', 
                             params_shape, 
                             tf.float32,
                             initializer=tf.constant_initializer(0.0, tf.float32))
      # scale
      gamma = tf.get_variable('gamma', 
                              params_shape, 
                              tf.float32,
                              initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        # 为每个通道计算均值、标准差
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
        # 新建或建立测试阶段使用的batch均值、标准差
        moving_mean = tf.get_variable('moving_mean', 
                                      params_shape, tf.float32,
                                      initializer=tf.constant_initializer(0.0, tf.float32),
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance', 
                                          params_shape, tf.float32,
                                          initializer=tf.constant_initializer(1.0, tf.float32),
                                          trainable=False)
        # 添加batch均值和标准差的更新操作(滑动平均)
        # moving_mean = moving_mean * decay + mean * (1 - decay)
        # moving_variance = moving_variance * decay + variance * (1 - decay)
        self._extra_train_ops.append(moving_averages.assign_moving_average(
                                                        moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
                                                        moving_variance, variance, 0.9))
      else:
        # 获取训练中积累的batch均值、标准差
        mean = tf.get_variable('moving_mean', 
                               params_shape, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32),
                               trainable=False)
        variance = tf.get_variable('moving_variance', 
                                   params_shape, tf.float32,
                                   initializer=tf.constant_initializer(1.0, tf.float32),
                                   trainable=False)
        # 添加到直方图总结
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)

      # BN层：((x-mean)/var)*gamma+beta
      y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y


  # 权重衰减，L2正则loss
  def _decay(self):
    costs = []
    # 遍历所有可训练变量
    for var in tf.trainable_variables():
      #只计算标有“DW”的变量
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    # 加和，并乘以衰减因子
    return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

  # 2D卷积
  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      # 获取或新建卷积核，正态随机初始化
      kernel = tf.get_variable(
              'DW', 
              [filter_size, filter_size, in_filters, out_filters],
              tf.float32, 
              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
      # 计算卷积
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  # leaky ReLU激活函数，泄漏参数leakiness为0就是标准ReLU
  def _relu(self, x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
  
  # 全连接层，网络最后一层
  def _fully_connected(self, x, out_dim):
    # 输入转换成2D tensor，尺寸为[N,-1]
    x = tf.reshape(x, [self.hps.batch_size, -1])
    # 参数w，平均随机初始化，[-sqrt(3/dim), sqrt(3/dim)]*factor
    w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    # 参数b，0值初始化
    b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
    # 计算x*w+b
    return tf.nn.xw_plus_b(x, w, b)

  # 全局均值池化
  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    # 在第2&3维度上计算均值，尺寸由WxH收缩为1x1
    return tf.reduce_mean(x, [1, 2])
