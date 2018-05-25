'''
Convolutional Neural Network model for PDF identification
'''

import tensorflow as tf
import numpy as np
import input_data

# PIXEL_DIMENSION_WIDTH = 1828
# PIXEL_DIMENSION_HEIGHT = 1306
PIXEL_DIMENSION_WIDTH = 270
PIXEL_DIMENSION_HEIGHT = 360
FIRST_CONV_LAYER_FILTERS = 32
DOWNSAMPLE_SIZE = 2
KERNEL_SIZE = 5
# DENSE_NEURON_NUM = 1024
DENSE_NEURON_NUM = 240
LOGITS_NEURON_NUM = 2   # Number of target class (tablePositive, tableNegative)

def cnn_pdf_model(features):

    # Input layer
    # [batch_size, image_height, image_width, channels]
    with tf.name_scope('reshape'):
        pdf_input = tf.reshape(features, [-1, PIXEL_DIMENSION_WIDTH, PIXEL_DIMENSION_HEIGHT, 1])
        print("input: ", pdf_input)

    # First convolutional layer
    with tf.name_scope('first_conv'):
        W_conv1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, 1, 32])
        b_conv1 = bias_variable([32])
        first_conv = tf.nn.relu(convolutional_2d_nn(pdf_input, W_conv1)+b_conv1)

    # First pooling layer
    with tf.name_scope('first_pool'):
        first_pool = max_pool_n(first_conv, DOWNSAMPLE_SIZE)

    # First dense layer
    with tf.name_scope('first_dense'):
        # first_pool_batch = first_pool.get_shape()[0]
        print(first_pool.get_shape())
        first_pool_width = tf.cast(first_pool.get_shape()[1], tf.int32)
        first_pool_height = tf.cast(first_pool.get_shape()[2], tf.int32)
        first_pool_channel = tf.cast(first_pool.get_shape()[3], tf.int32)

        # first_pool_width*first_pool_height*first_pool_channel
        # w_dense = weight_variable([914*653*32, DENSE_NEURON_NUM])
        w_dense = weight_variable([135 * 180 * 32, DENSE_NEURON_NUM])
        b_dense = bias_variable([DENSE_NEURON_NUM])
        pool_flat = tf.reshape(first_pool, [-1,first_pool_width*first_pool_height*first_pool_channel])
        dense = tf.nn.relu(tf.matmul(pool_flat,w_dense)+b_dense)

    # Dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        dropout = tf.nn.dropout(dense, keep_prob=keep_prob)

    # Logits layer
    with tf.name_scope('logits'):
        w_dense = weight_variable([DENSE_NEURON_NUM,LOGITS_NEURON_NUM])
        b_dense = bias_variable([LOGITS_NEURON_NUM])
        y_conv = tf.matmul(dropout, w_dense)+b_dense

    return y_conv, keep_prob

def convolutional_2d_nn(input, weight):
    float32_input = tf.cast(input, tf.float32)
    return tf.nn.conv2d(float32_input, weight, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_n(value, n):
    return tf.nn.max_pool(value, ksize=[1,n,n,1], strides=[1,n,n,1], padding='SAME')

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
