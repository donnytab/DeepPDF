'''
Convolutional Neural Network model for PDF identification
'''

import tensorflow as tf
import numpy as np

PIXEL_DIMENSION_WIDTH = 28
PIXEL_DIMENSION_HEIGHT = 28
FIRST_CONV_LAYER_FILTERS = 32
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




def convolutional_2d_nn(input, weight):
    return tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))