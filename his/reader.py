#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import numpy as np
import tensorflow as tf
from os.path import exists


# from common import IMAGE_HEIGHT, IMAGE_SIZE, IMAGE_WIDTH, CAPTCHA_LEN, CHAR_SET_LEN, NUM_LABELS


def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class CaptchaReader(object):
    """ """
    def __init__(self, IMAGE_WIDTH, IMAGE_HEIGHT, CAPTCHA_LEN, charset):

        # 初始化参数
        self.alpha = 1e-3
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.charset = charset
        self.CAPTCHA_LEN = CAPTCHA_LEN
        self.step = 0

        # 计算相关变量
        self.IMAGE_SIZE = self.IMAGE_WIDTH * self.IMAGE_HEIGHT
        self.CHAR_SET_LEN = len(charset)
        self.NUM_LABELS = self.CHAR_SET_LEN * self.CAPTCHA_LEN

        # 定义神经网络
        self.__initail_grapth()

        # 启动神经网络
        # self.session = tf.Session(graph=self.graph)
        # init = tf.initialize_all_variables()
        # init = tf.global_variables_initializer()
        # self.session.run(init)
        # tf.global_variables_initializer().run(session=self.session)

    def __initail_grapth(self):  # `cnn` up to now
        with tf.Graph().as_default() as graph:
            # Define the PlaceHolder
            x = tf.placeholder(tf.float32, shape=[None, self.IMAGE_SIZE])
            y = tf.placeholder(tf.float32, shape=[None, self.NUM_LABELS])
            keep_prob = tf.placeholder(tf.float32)

            x_image = tf.reshape(x, shape=[-1, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1])

            # First Convolutional Layer, input@(100, 40), output@(50, 20)
            conv_layer1_weight = weight_variable([5, 5, 1, 32])
            conv_layer1_bias = bias_variable([32])
            pool_layer1 = max_pool(
                tf.nn.relu(
                    conv2d(x_image, conv_layer1_weight) + conv_layer1_bias
                )
            )

            # Second Convolutional Layer, input@(50, 20), output@(25, 10)
            conv_layer2_weight = weight_variable([5, 5, 32, 64])
            conv_layer2_bias = bias_variable([64])
            pool_layer2 = max_pool(
                tf.nn.relu(
                    conv2d(pool_layer1, conv_layer2_weight) + conv_layer2_bias
                )
            )

            # Third Convolutional Layer, input@(25, 10), output@(13, 5)
            conv_layer3_weight = weight_variable([5, 5, 64, 64])
            conv_layer3_bias = bias_variable([64])
            pool_layer3 = max_pool(
                tf.nn.relu(
                    conv2d(pool_layer2, conv_layer3_weight) + conv_layer3_bias
                )
            )

            # Fully Connected Layer
            last_width = int(round(self.IMAGE_WIDTH / 8))
            last_height = int(round(self.IMAGE_HEIGHT / 8))
            # fc_layer_weight = weight_variable([13 * 5 * 64, 1024])
            fc_layer_weight = weight_variable([last_width * last_height * 64, 1024])
            fc_layer_bias = bias_variable([1024])

            # pool_layer3_flat = tf.reshape(pool_layer3, [-1, 13 * 5 * 64])
            pool_layer3_flat = tf.reshape(pool_layer3, [-1, last_width * last_height * 64])
            fc_layer = tf.nn.relu(tf.add(tf.matmul(pool_layer3_flat, fc_layer_weight), fc_layer_bias))

            # Dropout
            fc_layer_drop = tf.nn.dropout(fc_layer, keep_prob)

            # Readout Layer
            output_layer_weight = weight_variable([1024, self.NUM_LABELS])
            output_layer_bias = bias_variable([self.NUM_LABELS])

            y_conv = tf.add(tf.matmul(fc_layer_drop, output_layer_weight), output_layer_bias)

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_conv)
            )

            optimizer = tf.train.AdamOptimizer(self.alpha).minimize(loss)

            prediction = tf.argmax(tf.reshape(y_conv, [-1, self.CAPTCHA_LEN, self.CHAR_SET_LEN]), 2)
            correct = tf.argmax(tf.reshape(y, [-1, self.CAPTCHA_LEN, self.CHAR_SET_LEN]), 2)
            correct_prediction = tf.equal(prediction, correct)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            saver = tf.train.Saver(max_to_keep=2)

            self.x = x
            self.y = y
            self.optimizer = optimizer
            self.loss = loss
            self.keep_prob = keep_prob
            self.accuracy = accuracy
            self.prediction = prediction
            self.saver = saver
            self.graph = graph

    def imageToVertor(self, image):
        """ 将图片转化为向量表示 """
        image = image.convert("L")
        image = np.asarray(image)
        image = image.reshape([self.IMAGE_WIDTH * self.IMAGE_HEIGHT]) / 255
        return image

    def codeToVertor(self, code):
        """ 将验证码转化为向量表示 """
        labels = np.zeros([self.CAPTCHA_LEN, len(self.charset)])
        for i in range(self.CAPTCHA_LEN):
            labels[i, self.charset.index(code[i])] = 1
        return labels.reshape(len(self.charset) * self.CAPTCHA_LEN)

    def train(self, imageList, codeList):
        """ 训练 """
        filename = 'model.ckpt'
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            if exists(filename):
                self.saver.restore(session, filename)
            batch_data = np.array(map(self.imageToVertor, imageList))
            batch_labels = np.array(map(self.codeToVertor, codeList))
            self.step += 1
            _, loss = session.run(
                [self.optimizer, self.loss],
                feed_dict={
                    self.x: batch_data,
                    self.y: batch_labels,
                    self.keep_prob: 0.75
                }
            )
            self.saver.save(session, filename)
        print u'loss:', loss

    def predict(self, image):
        """ 识别 """

    def save(self, filename):
        """ 保存网络状态 """
        self.saver.save(self.session, filename, global_step=self.step)

    def load(self, filename):
        """ 加载网络状态 """
        self.saver.restore(self.session, filename)
