#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import os
from helper import create_model
from define import code_length


def main():

    for i, index in enumerate(range(code_length), 1):
        model_path = 'model/%s/' % index
        nodes_file_name = os.path.join(model_path, 'nodes.pk')
        if not os.path.exists(nodes_file_name):
            create_model(model_path)


if __name__ == "__main__":
    main()
