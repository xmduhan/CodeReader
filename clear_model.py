#!/usr/bin/env python
# encoding: utf-8
import os
import shutil


def main():
    model_path = 'model'
    to_remove = map(lambda x: os.path.join(model_path, x), os.listdir(model_path))
    for i in to_remove:
        if os.path.isdir(i):
            shutil.rmtree(i)


if __name__ == "__main__":
    main()
