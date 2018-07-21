# -*- coding: utf-8 -*-
"""
@author: tz_zs

MNIST 升级----mnist_inference.py
"""
import tensorflow as tf

# 定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 正则化损失
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    # 正则化损失
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights


# 神经网络的前向传播过程
def inference(input_tensor, regularizer):
    # 第一层
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

        # 第二层
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))

        layer2 = tf.matmul(layer1, weights) + biases

    return layer2  