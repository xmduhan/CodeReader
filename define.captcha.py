#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from captcha.image import ImageCaptcha
from string import digits
# from string import lowercase

width = 100
height = 60
code_length = 4
# charset = digits + lowercase
charset = digits

# 验证码图片化对象
captcha = ImageCaptcha(width=width, height=height)


def generate_image(code):
    """ 将验证码转化为图片 """
    return captcha.generate_image(code)


def get_code():
    """ 生成验证码 """
    return ''.join(map(
        lambda x: charset[x],
        np.random.randint(0, len(charset), code_length)))
