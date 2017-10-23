#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import os
import pickle
import pandas as pd
import tensorflow as tf
from collections import deque
from datetime import datetime
from helper import get_data
from helper import create_model
from helper import image_to_vector
from helper import code_to_vector
from define import code_length
from config import max_train_time
from config import stat_length
from config import accuracy_level
from config import cpu_to_use
tf.logging.set_verbosity(tf.logging.ERROR)


def main():

    for i, index in enumerate(range(code_length), 1):
        model_path = 'model/%s/' % index
        model_file_name = os.path.join(model_path, 'model')
        nodes_file_name = os.path.join(model_path, 'nodes.pk')
        if not os.path.exists(nodes_file_name):
            create_model(model_path)

        recent_accuracy = deque(maxlen=stat_length)
        graph = tf.Graph()
        config = tf.ConfigProto(intra_op_parallelism_threads=cpu_to_use)
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
            for j, step in enumerate(range(max_train_time), 1):
                begin_time = datetime.now()
                imageList, codeList = get_data(100)
                codeList = map(lambda x: x[index], codeList)
                x_data = map(image_to_vector, imageList)
                y_data = map(code_to_vector, codeList)
                _, l, a = session.run(
                    [optimizer, loss, accuracy],
                    feed_dict={x: x_data, y: y_data, keep_prob: .75})
                if step % 10 == 0:
                    saver.save(session, model_file_name)
                end_time = datetime.now()
                dt = end_time - begin_time
                recent_accuracy.append(a)
                mean_of_accuracy = pd.Series(recent_accuracy).mean()
                format_string = '[%d(%d/%d): %d/%d]: loss: %.2f, accuracy: %.2f, accuracy mean: %.2f(<%.2f?), time: %.2f'
                print format_string % (index, i, code_length, j, max_train_time, l, a, mean_of_accuracy, accuracy_level, dt.total_seconds())
                if len(recent_accuracy) == stat_length:
                    if mean_of_accuracy >= accuracy_level:
                        break


if __name__ == "__main__":
    main()
