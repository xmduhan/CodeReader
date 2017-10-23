#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import os
import pickle
from helper import image_to_vector
from define import code_length
from define import charset
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
path = os.path.dirname(os.path.realpath(__file__))
# print path


modelList = []
for index in range(code_length):
    model_path = os.path.join(path, 'model/%s/' % index)
    # print model_path
    model_file_name = os.path.join(model_path, 'model')
    nodes_file_name = os.path.join(model_path, 'nodes.pk')

    model = {}
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with session.graph.as_default():
        saver = tf.train.import_meta_graph(model_file_name + '.meta')
        saver.restore(session, model_file_name)
        nodes = pickle.load(open(nodes_file_name, 'rU'))
        x = session.graph.get_tensor_by_name(nodes['x'])
        keep_prob = session.graph.get_tensor_by_name(nodes['keep_prob'])
        prediction = session.graph.get_tensor_by_name(nodes['prediction'])
        model['session'] = session
        model['x'] = x
        model['keep_prob'] = keep_prob
        model['prediction'] = prediction
        modelList.append(model)


def read_char(image, index):
    """ """
    # 读取模型
    model = modelList[index]
    session = model['session']
    x = model['x']
    keep_prob = model['keep_prob']
    prediction = model['prediction']

    # 进行预测
    with session.graph.as_default():
        imageList = [image]
        x_data = map(image_to_vector, imageList)
        p = session.run(prediction, feed_dict={x: x_data, keep_prob: 1})
        result = charset[p[0][0]]
    return result


def read_code(image):
    """ """
    return ''.join(map(lambda index: read_char(image, index), range(code_length)))
