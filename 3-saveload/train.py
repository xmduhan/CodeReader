#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import os
import pickle
import pandas as pd
import tensorflow as tf
from collections import deque
from helper import getData
from helper import create_model
from helper import imageToVertor
from helper import codeToVertor
from define import codeLength


def main():
    stat_length = 30
    accuracy_level = .99

    for i, index in enumerate(range(codeLength), 1):
        model_path = 'model/%s/' % index
        model_file_name = os.path.join(model_path, 'model')
        nodes_file_name = os.path.join(model_path, 'nodes.pk')
        if not os.path.exists(nodes_file_name):
            create_model(model_path)

        recent_accuracy = deque(maxlen=stat_length)
        graph = tf.Graph()
        config = tf.ConfigProto(intra_op_parallelism_threads=2)
        session = tf.Session(graph=graph, config=config)
        with session.graph.as_default():

            # 导入模型定义
            saver = tf.train.import_meta_graph(model_file_name + '.meta')
            saver.restore(session, model_file_name)
            nodes = pickle.load(open(nodes_file_name, "rU"))
            x = session.graph.get_tensor_by_name(nodes['x'])
            y = session.graph.get_tensor_by_name(nodes['y'])
            keep_prob = session.graph.get_tensor_by_name(nodes['keep_prob'])
            loss = session.graph.get_tensor_by_name(nodes['loss'])
            accuracy = session.graph.get_tensor_by_name(nodes['accuracy'])
            optimizer = session.graph.get_operation_by_name(nodes['optimizer'])

            # 训练模型
            for step in range(20000):
                imageList, codeList = getData(100)
                codeList = map(lambda x: x[index], codeList)
                x_data = map(imageToVertor, imageList)
                y_data = map(codeToVertor, codeList)
                _, l, a = session.run(
                    [optimizer, loss, accuracy],
                    feed_dict={x: x_data, y: y_data, keep_prob: .75})
                if step % 10 == 0:
                    saver.save(session, model_file_name)
                recent_accuracy.append(a)
                mean_of_accuracy = pd.Series(recent_accuracy).mean()
                format_string = '[%d(%d/%d):%d]: loss: %f, accuracy: %f, accuracy mean: %f(<%.2f?)'
                print format_string % (index, i, codeLength, step, l, a, mean_of_accuracy, accuracy_level)
                if len(recent_accuracy) == stat_length:
                    if mean_of_accuracy >= accuracy_level:
                        break


if __name__ == "__main__":
    main()
