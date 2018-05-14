'''
Convolutional Neural Network model for PDF identification
'''

import tensorflow as tf
import numpy as np

PIXEL_DIMENSION_WIDTH = 28
PIXEL_DIMENSION_HEIGHT = 28
FIRST_CONV_LAYER_FILTERS = 32
DOWNSAMPLE_SIZE = 2
KERNEL_SIZE = 5

def cnn_pdf_model(features):

    # Input layer
    with tf.name_scope('reshape'):
        pdf_input = tf.reshape(features, [-1, PIXEL_DIMENSION_WIDTH, PIXEL_DIMENSION_HEIGHT, 1])

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
        w_dense = weight_variable([7*7*64,1024])
        b_dense = bias_variable([1024])
        pool_flat = tf.reshape(first_pool, [-1,7*7*64])
        dense = tf.nn.relu(tf.matmul(pool_flat,w_dense)+b_dense)

    # Dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        dropout = tf.nn.dropout(dense, keep_prob=keep_prob)

    # Logits layer
    with tf.name_scope('logits'):
        w_dense = weight_variable([1024,10])
        b_dense = bias_variable([10])
        y_conv = tf.matmul(dropout, w_dense)+b_dense

    return y_conv, keep_prob

def convolutional_2d_nn(input, weight):
    return tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_n(value, n):
    return tf.nn.max_pool(value, ksize=[1,n,n,1], strides=[1,n,n,1], padding='SAME')

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))