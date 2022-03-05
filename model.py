from typing import Dict, List, Optional

import tensorflow as tf
from tensorflow import Tensor
# noinspection PyPackageRequirements
from keras.layers import (
  Conv2D,
  BatchNormalization,
  SpatialDropout2D,
  GlobalAveragePooling2D,
  LeakyReLU,
  Dense,
  Add
)


class ResidualBlock(tf.keras.layers.Layer):
  def __init__(self,
               num_of_filters: int = 16,
               final_relu: bool = True,
               **kwargs):

    super(ResidualBlock, self).__init__(**kwargs)
    self.num_of_filters = num_of_filters
    self.final_relu = final_relu
    self.conv_1 = None
    self.conv_2 = None
    self.relu_1 = None
    self.relu_2 = None
    self.norm_1 = None
    self.norm_2 = None
    if self.final_relu:
      self.add_1 = None

  def build(self, input_shape: List) -> None:
    self.conv_1 = Conv2D(filters=self.num_of_filters,
                         kernel_size=(1, 1),
                         input_shape=input_shape)

    self.norm_1 = BatchNormalization()
    self.relu_1 = LeakyReLU()
    self.conv_2 = Conv2D(filters=self.num_of_filters,
                         kernel_size=3,
                         padding="same")
    self.norm_2 = BatchNormalization()
    self.add_1 = Add()
    if self.final_relu:
      self.relu_2 = LeakyReLU()

  def call(self, inputs: Tensor) -> Tensor:
    layer = self.conv_1(inputs)
    layer = self.norm_1(layer)
    layer = self.relu_1(layer)
    layer = self.conv_2(layer)
    layer = self.norm_2(layer)
    layer = self.add_1([layer, inputs])
    if self.final_relu:
      layer = self.relu_2(layer)

    return layer

  def get_config(self) -> Dict:
    config = super(ResidualBlock, self).get_config()
    config.update({"num_of_filters": self.num_of_filters,
                   "norm_type": self.norm_type,
                   "final_relu": self.final_relu})

    return config


def get_model(kernel_sizes: Optional[List[int]] = None,
              stride_sizes: Optional[List[int]] = None,
              pad_inputs: bool = True,
              dropout_rate: float = 0.2) -> tf.keras.Model:
  if kernel_sizes is None:
    kernel_sizes = [3, 3, 3]
  if stride_sizes is None:
    stride_sizes = [2, 2, 2]

  relu = LeakyReLU()
  input_layer = tf.keras.Input(shape=(28, 28, 1), name="input_image")
  if pad_inputs:
    x = tf.pad(input_layer, [[0, 0], [2, 2], [2, 2], [0, 0]], "SYMMETRIC")
  else:
    x = input_layer

  x = Conv2D(1,
             kernel_size=kernel_sizes[0],
             padding="valid" if pad_inputs else "same",
             strides=stride_sizes[1])(x)
  x = SpatialDropout2D(dropout_rate)(x)
  x = BatchNormalization()(x)
  x = relu(x)

  residual_block_filters = 128
  for _ in range(5):
    # noinspection PyCallingNonCallable
    x = ResidualBlock(residual_block_filters, False)(x)
    x = SpatialDropout2D(dropout_rate)(x)

  x = GlobalAveragePooling2D()(x)
  x = Dense(10, activation="softmax")(x)

  return tf.keras.Model(input_layer, x, name="fashion_mnist_classifier")
