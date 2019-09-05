# 简单的进行数据分析

import fer_config as config
import fer_generator
import pandas as pd
import numpy as np
from PIL import Image
#1.数据均衡问题
#可以考虑让这个分类的loss乘以10，但是如何实现？batch内又不都是同类。不是batch的话，怎么乘以权重，难道一个一个的计算。
data = pd.read_csv(config.data_path)
print(type(data))
images = data['pixels']
labels = data['emotion']
usages = data['Usage']
print(type(images))

cnt = [0] * 7

for i in range(len(images)):
    if usages[i] == 'Training':
        # print(labels[i])
        cnt[labels[i]] += 1
        # print(images[i])

print(cnt)


print((images[0]).split(' '))
print(np.array((images[0]).split(' ')))
arr = np.array((images[0]).split(' '),dtype=int)
print(arr)
print(arr.shape)
# print(arr.reshape(newshape=(28,28)))
reshaped = np.reshape(arr,newshape=(48,48))
print(reshaped)
print(arr.reshape==np.reshape)
print(type(arr))

img = Image.fromarray(reshaped)
img.show()


