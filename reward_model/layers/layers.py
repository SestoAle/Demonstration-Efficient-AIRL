import tensorflow as tf
from tensorforce import util
from math import sqrt

## Layers
def linear(input, inner_size, name='linear', bias=True, activation=None, init=None):
    with tf.compat.v1.variable_scope(name):
        lin = tf.compat.v1.layers.dense(input, inner_size, name=name, activation=activation, use_bias=bias,
                                        kernel_initializer=init)
        return lin

def conv_layer_2d(input, filters, kernel_size, strides=(1, 1), padding="SAME", name='conv',
                  activation=None,
                  bias=True):
    with tf.compat.v1.variable_scope(name):
        conv = tf.compat.v1.layers.conv2d(input, filters, kernel_size, strides, padding=padding, name=name,
                                          activation=activation, use_bias=bias)
        return conv

def embedding(input, indices, size, name='embs'):
    with tf.compat.v1.variable_scope(name):
        shape = (indices, size)
        stddev = min(0.1, sqrt(2.0 / (util.product(xs=shape[:-1]) + shape[-1])))
        initializer = tf.random.normal(shape=shape, stddev=stddev, dtype=tf.float32)
        W = tf.Variable(
            initial_value=initializer, trainable=True, validate_shape=True, name='W',
            dtype=tf.float32, shape=shape
        )
        return tf.nn.tanh(tf.compat.v1.nn.embedding_lookup(params=W, ids=input, max_norm=None))