#!/usr/bin/env python
# encoding: utf-8
from __future__ import division
import os
import math
import pickle
import numpy as np
from os import path
import tensorflow as tf
from define import width
from define import height
from define import charset
from define import get_code
from define import generate_image


def get_data(n=10):
    """ 获取数据集  """
    codeList = [get_code() for _ in range(n)]
    imageList = map(generate_image, codeList)
    return imageList, codeList


def image_to_vector(image):
    """ 将图片转化为向量表示 """
    width = image.width
    height = image.height
    image = image.convert("L")
    image = np.asarray(image)
    image = image.reshape([width * height]) / 255
    return image


def code_to_vector(code):
    """ 将验证码转化为向量表示 """
    code_length = len(code)
    labels = np.zeros([code_length, len(charset)])
    for i in range(code_length):
        labels[i, charset.index(code[i])] = 1
    return labels.reshape(len(charset) * code_length)


def weight_variable(shape):
    """  """
    initial = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    """ """
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(x, W):
    """ """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_model(model_path, alpha=1e-3):
    """
    width
    height
    charset
    """
    # 计算输入输出的长宽度
    image_size = width * height
    last_width = int(math.ceil(width / 8))
    last_height = int(math.ceil(height / 8))

    # 保存模型结构
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_file_name = path.join(model_path, 'model')

    graph = tf.Graph()
    with graph.as_default():
        # 定义输入输出
        x = tf.placeholder(tf.float32, shape=[None, image_size])
        y = tf.placeholder(tf.float32, shape=[None, len(charset) * 1])  # 一次识别一个字符
        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(x, shape=[-1, width, height, 1])

        # 定义第一层卷积
        conv_layer1_weight = weight_variable([5, 5, 1, 32])
        conv_layer1_bias = bias_variable([32])
        pool_layer1 = max_pool(tf.nn.relu(conv2d(x_image, conv_layer1_weight) + conv_layer1_bias))

        # 定义第二层卷积
        conv_layer2_weight = weight_variable([5, 5, 32, 64])
        conv_layer2_bias = bias_variable([64])
        pool_layer2 = max_pool(tf.nn.relu(conv2d(pool_layer1, conv_layer2_weight) + conv_layer2_bias))

        # 定义第三层卷积
        conv_layer3_weight = weight_variable([5, 5, 64, 64])
        conv_layer3_bias = bias_variable([64])
        pool_layer3 = max_pool(tf.nn.relu(conv2d(pool_layer2, conv_layer3_weight) + conv_layer3_bias))

        # 定义全连接层
        fc_layer_weight = weight_variable([last_width * last_height * 64, 1024])
        fc_layer_bias = bias_variable([1024])
        pool_layer3_flat = tf.reshape(pool_layer3, [-1, last_width * last_height * 64])
        fc_layer = tf.nn.relu(tf.add(tf.matmul(pool_layer3_flat, fc_layer_weight), fc_layer_bias))

        # Dropout层
        fc_layer_drop = tf.nn.dropout(fc_layer, keep_prob)

        # Readout层(输出层)
        output_layer_weight = weight_variable([1024, len(charset) * 1])  # 一次识别一个字符
        output_layer_bias = bias_variable([len(charset) * 1])  # 一次识别一个字符
        y_conv = tf.add(tf.matmul(fc_layer_drop, output_layer_weight), output_layer_bias)

        # 定义输出函数
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_conv))
        optimizer = tf.train.AdamOptimizer(alpha).minimize(loss)
        prediction = tf.argmax(tf.reshape(y_conv, [-1, 1, len(charset)]), 2)  # 一次识别一个字符
        correct = tf.argmax(tf.reshape(y, [-1, 1, len(charset)]), 2)  # 一次识别一个字符
        correct_prediction = tf.equal(prediction, correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 保存模型
    # session = tf.Session(graph=graph)
    # session.run(tf.global_variables_initializer())
    # saver = tf.train.Saver(max_to_keep=1)
    # saver.save(session, model_file_name)

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(session, model_file_name)

    # 保存名称字典
    nodes = {
        'x': x.name,
        'y': y.name,
        'prediction': prediction.name,
        'keep_prob': keep_prob.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'accuracy': accuracy.name,
    }
    nodes_file_name = path.join(model_path, 'nodes.pk')
    pickle.dump(nodes, open(nodes_file_name, 'wb'))
