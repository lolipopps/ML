# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:02:19 2017

@author: osT
"""
import random
import csv
import tensorflow as tf
import numpy as np

#定义神经网络的输入输出结点，每个样本为1*243维，以及输出分类结果
INPUT_NODE=242
OUTPUT_NODE=30

#定义两层隐含层的神经网络，一层300个结点，一层100个结点
LAYER1_NODE=120
LAYER2_NODE=20

#定义学习率，学习率衰减速度，正则系数，训练调整参数的次数以及平滑衰减率
LEARNING_RATE_BASE=0.005
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.001
TRAINING_STEPS=100000
MOVING_AVERAGE_DECAY=0.99


#定义整个神经网络的结构，也就是向前传播的过程，avg_class为平滑可训练量的类，不传入则不使用平滑
def inference(input_tensor,avg_class,w1,b1,w2,b2,w3,b3):
    if avg_class==None:
        #第一层隐含层，输入与权重矩阵乘后加上常数传入激活函数作为输出
        layer1=tf.nn.relu(tf.matmul(input_tensor,w1)+b1)
        #第二层隐含层，前一层的输出与权重矩阵乘后加上常数作为输出
        layer2=tf.nn.relu(tf.matmul(layer1,w2)+b2)
        #返回 第二层隐含层与权重矩阵乘加上常数作为输出
        return tf.matmul(layer2,w3)+b3
    else:
        #avg_class.average()平滑训练变量，也就是每一层与上一层的权重
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(w1))+avg_class.average(b1))
        layer2=tf.nn.relu(tf.matmul(layer1,avg_class.average(w2))+avg_class.average(b2))
        return tf.matmul(layer2,avg_class.average(w3))+avg_class.average(b3)
def get_fromfile():
    lists = []
    file = open("./data/cheng.txt",'r')
    for data in file.readlines():
        list = [i for i in data.replace("\n","").split(" ")]
        lists.append(list)
    random.shuffle(lists)
    file.close()
    return np.array(lists)

def get_rangefile():
    lists = []
    file = open("./data/cheng.txt",'r')
    for data in file.readlines():
        list = [i for i in data.replace("\n","").split(" ")]
        lists.append(list)
    file.close()
    return np.array(lists)

def pre():
    #定义输出数据的地方，None表示无规定一次输入多少训练样本,y_是样本标签存放的地方
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='x-input')
    #y_=tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name='y-input')

    #依次定义每一层与上一层的权重，这里用随机数初始化，注意shape的对应关系
    with tf.name_scope('weights1'):
        w1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAYER1_NODE],stddev=0.1))
        tf.summary.histogram("w1" + "/weights", w1)  # 可视化观看变量
        b1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
        tf.summary.histogram("b1" + "/weights", b1)  # 可视化观看变量
    with tf.name_scope('weights2'):

        w2=tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE,LAYER2_NODE],stddev=0.1))
        tf.summary.histogram("w2" + "/weights", w2)  # 可视化观看变量
        b2=tf.Variable(tf.constant(0.1,shape=[LAYER2_NODE]))
        tf.summary.histogram("b2" + "/weights", b2)  # 可视化观看变量

    w3=tf.Variable(tf.truncated_normal(shape=[LAYER2_NODE,OUTPUT_NODE],stddev=0.1))
    b3=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #输出向前传播的结果
    y=inference(x,None,w1,b1,w2,b2,w3,b3)

    #每训练完一次就会增加的变量
    global_step=tf.Variable(0,trainable=False)

    #定义平滑变量的类，输入为平滑衰减率和global_stop使得每训练完一次就会使用平滑过程
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    #将平滑应用到所有可训练的变量，即trainable=True的变量
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    #输出平滑后的预测值
    average_y=inference(x,variable_averages,w1,b1,w2,b2,w3,b3)
    #定义交叉熵和损失函数，但为什么传入的是label的arg_max(),就是对应分类的下标呢，我们迟点再说
    cross_entropy=tf.nn.softmax(average_y, name='prob')
    max_pro = tf.argmax(cross_entropy, 1)
    saver_path = './save/model2.ckpt'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, saver_path)  #
        data_x = get_rangefile()
        for i in range(len(data_x)):
            labels = data_x[i:i+1, :242]
            label = data_x[i:i+1, 242:-1]
            label = label[0]
            print("原始为:",label ,"预测为",sess.run(max_pro+1, feed_dict={x:labels}))
pre()