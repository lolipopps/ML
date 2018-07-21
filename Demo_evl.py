# -*- coding: utf-8 -*-
"""
@author: tz_zs

MNIST 升级----mnist_train.py
"""
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
# 加载函数
import mnist_inference

# 配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存路径和文件名
MODEL_SAVE_PATH = "/path/to/model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    # 定义输入输出的placeholder
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    # 定义正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 使用前向传播
    y = mnist_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 优化算法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

        # 持久化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 运行
            _, loss_valuue, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            # 每1000轮保存一次模型
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_valuue))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()