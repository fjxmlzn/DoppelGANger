import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


def linear(input_, output_size, scope_name="linear"):
    with tf.compat.v1.variable_scope(scope_name):
        input_ = tf.reshape(
            input_,
            [-1, np.prod(input_.get_shape().as_list()[1:])])
        output = tf.compat.v1.layers.dense(
            input_,
            output_size)
        return output


def flatten(input_, scope_name="flatten"):
    with tf.compat.v1.variable_scope(scope_name):
        output = tf.reshape(
            input_,
            [-1, np.prod(input_.get_shape().as_list()[1:])])
        return output


class batch_norm(object):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.compat.v1.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        # I removed updates_collections=None since I didn't find the alternative
        return tf.keras.layers.BatchNormalization(momentum=self.momentum,
                                                  epsilon=self.epsilon,
                                                  scale=True,
                                                  trainable=train,
                                                  name=self.name)(x)


class layer_norm(object):
    def __init__(self, name="layer_norm"):
        self.name = name

    def __call__(self, x):
        return tf.keras.layers.LayerNormalization(name=self.name)(x)


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    with tf.compat.v1.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.compat.v1.get_variable(
            'w',
            [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
            initializer=tf.compat.v1.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(
                input_,
                w,
                output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(
                input_,
                w,
                output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

        biases = tf.compat.v1.get_variable(
            'biases',
            [output_shape[-1]],
            initializer=tf.compat.v1.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)

        return deconv


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable(
            'w',
            [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(
            input=input_,
            filters=w,
            strides=[1, d_h, d_w, 1],
            padding='SAME')

        biases = tf.compat.v1.get_variable(
            'biases', [output_dim], initializer=tf.compat.v1.constant_initializer(0.0))
        conv = tf.reshape(
            tf.nn.bias_add(conv, biases),
            [-1] + conv.get_shape().as_list()[1:])

        return conv


def lrelu(x, leak=0.2, name="lrelu"):
    # Code from:
    # https://github.com/carpedm20/DCGAN-tensorflow
    return tf.maximum(x, leak * x)
