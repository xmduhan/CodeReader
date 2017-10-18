

### .生成验证码定义文件(define.py)
``` python
# 图片宽度
width = 100

# 图片高度
height = 50

# 验证码的长度
code_length = 4

# 验证码可能出现的字符
charset = '0123456789'

# 获取随机验证码
def get_code():
    ...

# 通过验证码生成验证码图片
def generate_image(code):
    ...
```
具体可以参考文件: define.captcha.py和define.kaptcha.py

### .生成模型
``` shell
./create_model.py
```

### .编辑配置文件(config.py)
``` shell
vi config.py
```
设置模型相关训练参数

### .训练
``` shell
./train_model.py
```

### .测试
打开test.ipynb测试识别效果
