#!/usr/bin/env python
# encoding: utf-8

import tempfile
from string import digits
from string import lowercase
from itertools import product
from PIL import Image
import py4j
from py4j.java_gateway import JavaGateway, GatewayParameters
from random import sample

width = 223
height = 50
code_length = 5
charset = digits + lowercase

# 验证码图片化对象
port = py4j.java_gateway.launch_gateway(classpath='/usr/lib/jvm/lib/kaptcha-2.3.2.jar')
gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
constants = gateway.jvm.com.google.code.kaptcha.Constants
ImageIO = gateway.jvm.javax.imageio.ImageIO
filename = tempfile.mktemp(suffix='.jpg')
fontSizeList = ['39', '40', '41', '42', '43', '44', '45']
fontNameList = [
    'Ubuntu Light',
    'Ubuntu Light Italic',
    'Ubuntu Regular',
    'Ubuntu Regular Italic',
    'Lato-Hairline',
    'lmroman8-italic',
    'lmmonocaps10-regular',
    'Loma-Oblique',
    'Norasi-Oblique',
    'Umpush-Light',
]
kaptchaList = []

for fontSize, fontName in product(fontSizeList, fontNameList):
    properties = gateway.jvm.java.util.Properties()
    properties.put(constants.KAPTCHA_IMAGE_WIDTH, '223')
    properties.put(constants.KAPTCHA_IMAGE_HEIGHT, '50')
    properties.put(constants.KAPTCHA_TEXTPRODUCER_FONT_SIZE, fontSize)
    properties.put(constants.KAPTCHA_TEXTPRODUCER_FONT_NAMES, fontName)
    properties.put(constants.KAPTCHA_BORDER, 'no')
    kaptchaConfig = gateway.jvm.com.google.code.kaptcha.util.Config(properties)
    kaptcha = gateway.jvm.com.google.code.kaptcha.impl.DefaultKaptcha()
    kaptcha.setConfig(kaptchaConfig)
    kaptchaList.append(kaptcha)


def get_code():
    """ 生成验证码 """
    return kaptcha.createText()


def generate_image(code):
    """ 将验证码转化为图片 """
    kaptcha = sample(kaptchaList, 1)[0]
    image = kaptcha.createImage(code)
    f = gateway.jvm.java.io.File(filename)
    ImageIO.write(image, 'JPG', f)
    image = Image.open(filename)
    return image
