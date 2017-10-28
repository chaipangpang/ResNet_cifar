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

"""CIFAR dataset input module.
"""

import tensorflow as tf

def build_input(dataset, data_path, batch_size, mode):
  """Build CIFAR image and labels.

  Args:
    dataset(数据集): Either 'cifar10' or 'cifar100'.
    data_path(数据集路径): Filename for data.
    batch_size: Input batch size.
    mode(模式）: Either 'train' or 'eval'.
  Returns:
    images(图片): Batches of images. [batch_size, image_size, image_size, 3]
    labels(类别标签): Batches of labels. [batch_size, num_classes]
  Raises:
    ValueError: when the specified dataset is not supported.
  """
  
  # 数据集参数
  image_size = 32
  if dataset == 'cifar10':
    label_bytes = 1
    label_offset = 0
    num_classes = 10
  elif dataset == 'cifar100':
    label_bytes = 1
    label_offset = 1
    num_classes = 100
  else:
    raise ValueError('Not supported dataset %s', dataset)

  # 数据读取参数
  depth = 3
  image_bytes = image_size * image_size * depth
  record_bytes = label_bytes + label_offset + image_bytes

  # 获取文件名列表
  data_files = tf.gfile.Glob(data_path)
  # 文件名列表生成器
  file_queue = tf.train.string_input_producer(data_files, shuffle=True)
  # 文件名列表里读取原始二进制数据
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, value = reader.read(file_queue)

  # 将原始二进制数据转换成图片数据及类别标签
  record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
  label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
  # 将数据串 [depth * height * width] 转换成矩阵 [depth, height, width].
  depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]),
                           [depth, image_size, image_size])
  # 转换维数：[depth, height, width]转成[height, width, depth].
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  if mode == 'train':
    # 增减图片尺寸
    image = tf.image.resize_image_with_crop_or_pad(
                        image, image_size+4, image_size+4)
    # 随机裁剪图片
    image = tf.random_crop(image, [image_size, image_size, 3])
    # 随机水平翻转图片
    image = tf.image.random_flip_left_right(image)
    # 逐图片做像素值中心化(减均值)
    image = tf.image.per_image_standardization(image)

    # 建立输入数据队列(随机洗牌)
    example_queue = tf.RandomShuffleQueue(
        # 队列容量
        capacity=16 * batch_size,
        # 队列数据的最小容许量
        min_after_dequeue=8 * batch_size,
        dtypes=[tf.float32, tf.int32],
        # 图片数据尺寸，标签尺寸
        shapes=[[image_size, image_size, depth], [1]])
    # 读线程的数量
    num_threads = 16
  else:
    # 获取测试图片，并做像素值中心化
    image = tf.image.resize_image_with_crop_or_pad(
                        image, image_size, image_size)
    image = tf.image.per_image_standardization(image)

    # 建立输入数据队列(先入先出队列）
    example_queue = tf.FIFOQueue(
        3 * batch_size,
        dtypes=[tf.float32, tf.int32],
        shapes=[[image_size, image_size, depth], [1]])
    # 读线程的数量
    num_threads = 1

  # 数据入队操作
  example_enqueue_op = example_queue.enqueue([image, label])
  # 队列执行器
  tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
      example_queue, [example_enqueue_op] * num_threads))

  # 数据出队操作，从队列读取Batch数据
  images, labels = example_queue.dequeue_many(batch_size)
  # 将标签数据由稀疏格式转换成稠密格式
  # [ 2,       [[0,1,0,0,0]
  #   4,        [0,0,0,1,0]  
  #   3,   -->  [0,0,1,0,0]    
  #   5,        [0,0,0,0,1]
  #   1 ]       [1,0,0,0,0]]
  labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  labels = tf.sparse_to_dense(
                  tf.concat(values=[indices, labels], axis=1),
                  [batch_size, num_classes], 1.0, 0.0)

  #检测数据维度
  assert len(images.get_shape()) == 4
  assert images.get_shape()[0] == batch_size
  assert images.get_shape()[-1] == 3
  assert len(labels.get_shape()) == 2
  assert labels.get_shape()[0] == batch_size
  assert labels.get_shape()[1] == num_classes

  # 添加图片总结
  tf.summary.image('images', images)
  return images, labels
