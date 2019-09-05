#测试数据增强。
#思路：从哪一步做增强？目前有三步：csv拆成三个集合，csv变图片，图片变tfrecord。
#如果是tfrecord读出来，decode之前，可否操作？
#如果是存tfrecord之前处理，怎么处理？所有图片都做一次？数据翻倍？

#如果是读取tfrecord之后再翻转，势必影响效率？而且是以tensor和batch的形式存在，如何操作也是问题
import fer_config as config
import fer_generator
import tensorflow as tf
import fer_forward
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

flip_control = tf.random_uniform(shape=[1])#如何随机，用tf.cond！！
rotate_control = tf.random_uniform(shape=[1])
rotate_angle = tf.random_uniform(shape=[1],minval=-0.5,maxval=0.5,dtype=tf.float32)#我自己测0.5还可接受，因为有些图本来就偏，不能偏太大。
test_batch_size = 2
#read_tfRecord已经是生产者结构了，直接取不出来。必须当tensor。
img_batch,label_batch=fer_generator.get_tfrecord(test_batch_size,config.tfRecord_test_mini,data_augment=False)#mini测试集

reshaped_img_batch = tf.reshape(img_batch,shape=[test_batch_size,config.img_width,config.img_height,fer_forward.NUM_CHANNELS])

# todo tf自带了random前缀的接口 random_flip_left_right，不支持batch。。。。todo 如何实现batch内不同样本分别随机？先用全体统一随机凑合一下。
# flipped_img_batch = tf.image.flip_left_right(reshaped_img_batch)
flipped_img_batch = tf.cond(flip_control[0]>=0.5,lambda:tf.image.flip_left_right(reshaped_img_batch),lambda:reshaped_img_batch)
#tf.reverse可以达到同样效果，选对dim就好了
# flipped_img_batch = reshaped_img_batch#短路，测试用
rotated_img_batch = tf.cond(rotate_control[0]>=0.0,lambda:tf.contrib.image.rotate(flipped_img_batch,rotate_angle[0],interpolation = 'BILINEAR'),lambda:flipped_img_batch)#interpolation:  "NEAREST", "BILINEAR".
print('a')
with tf.Session() as sess:
    print(rotate_angle)
    print('angle:',sess.run(rotate_angle))

    # 创建一个线程协调器
    coord = tf.train.Coordinator()
    # 启动入队线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    xs, ys = sess.run([rotated_img_batch, label_batch])
    print('ys:',ys)
    print('img_batch.shape:',img_batch.shape)
    reshape_xs = np.reshape(xs, (test_batch_size,
                                 config.img_width,
                                 config.img_height,
                                 fer_forward.NUM_CHANNELS))
    print(type(reshape_xs))#是numpy类型的，可以直接打印出来看，或者直接翻转，都能办到。

    print(reshape_xs.shape)
    print(reshape_xs[0].shape)
    print(reshape_xs[0].dtype)
    # reshape_xs[0].dtype = np.float64

    #现在是0~1的float数据，需要转换类型和像素值才能观看。
    # 做这个过程的逆：
    # img = tf.cast(img, tf.float32) * (1. / 255)
    # label = tf.cast(features['label'], tf.float32)
    reshape_xs = reshape_xs * 255
    reshape_xs = reshape_xs.astype(np.int32)#reshape_xs.dtype = np.int32#这种赋值是不行的

    reshape_xs.resize((test_batch_size,config.img_width,config.img_height))#去掉channel的维度
    print('resized shape:',reshape_xs.shape)

    for i in range(test_batch_size):
        pic1 = Image.fromarray(reshape_xs[i])
        pic1.show()
        # pic1_transposed = pic1.transpose(Image.FLIP_LEFT_RIGHT)#左右翻转
        # pic1_transposed.show()
        # pic1_rotated = pic1.rotate(30)#效果不好？有黑边
        # pic1_rotated.show()
        # pic1.save('pic_saved'+ '_' + str(i) + '.png','png')


    coord.request_stop()
    coord.join(threads)




