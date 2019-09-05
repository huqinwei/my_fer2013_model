# coding:utf-8
import time
import tensorflow as tf
import fer_forward
import backward
import fer_generator
import numpy as np
import fer_config as config
import os
from PIL import Image
import matplotlib.pyplot as plt
#The data consists of 48x48 pixel grayscale images of faces.
#The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.
# The task is to categorize each face based on the emotion shown in the facial expression
# in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

prediction_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
print(prediction_dict[0])
print(type(prediction_dict[1]))

def test():

    # 实例化一个数据流图并作为整个 tensorflow 运行环境的默认图
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [1, config.img_width,
                                    config.img_height, fer_forward.NUM_CHANNELS])
        tf.summary.image('img_input',x)#记录输入图片
        y_ = tf.placeholder(tf.float32, [None, fer_forward.OUTPUT_NODE])

        prob = tf.placeholder(tf.float32)
        bn_training = tf.placeholder(tf.bool)
        # y = fer_forward.forward(x, keep_prob=prob)
        y,return_dict = fer_forward.forward(x,keep_prob=prob,bn_enable=True,bn_training=bn_training)

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()#生成ema替代原变量的映射关系。


        ##image是没有办法输出所有channel的（只有1、3、4三种支持），除非用循环把所有维度都分别丢进去吧。
        for i in range(return_dict['relu1_output'].shape[-1]):
            tf.summary.image('relu1/relu1_channel_'+str(i),
                         tf.expand_dims(input=return_dict['relu1_output'][:, :, :, i], axis=-1))  # 记录输出

        for i in range(return_dict['relu2_output'].shape[-1]):
            tf.summary.image('relu2/relu2_channel_' + str(i),
                             tf.expand_dims(input=return_dict['relu2_output'][:, :, :, i], axis=-1))
        for i in range(return_dict['relu3_output'].shape[-1]):
            tf.summary.image('relu3/relu3_channel_'+str(i),
                         tf.expand_dims(input=return_dict['relu3_output'][:, :, :, i], axis=-1))
        for i in range(return_dict['pool1_output'].shape[-1]):
            tf.summary.image('pool1/pool1_output'+str(i),
                         tf.expand_dims(input=return_dict['pool1_output'][:, :, :, i], axis=-1))
        for i in range(return_dict['pool2_output'].shape[-1]):
            tf.summary.image('pool2/pool2_output'+str(i),
                         tf.expand_dims(input=return_dict['pool2_output'][:, :, :, i], axis=-1))
        for i in range(return_dict['pool3_output'].shape[-1]):
            tf.summary.image('pool3/pool3_output'+str(i),
                         tf.expand_dims(input=return_dict['pool3_output'][:, :, :, i], axis=-1))


        print('yyyyyyyyyyyy:',y)
        tf.summary.histogram('y/pic',y[0])

        merged = tf.summary.merge_all()


        loader = tf.train.Saver(ema_restore)

        prediction = tf.argmax(y, 1)


        # 使用真实图片进行预测，包括各种处理。
        samples_column = 4
        samples_row = 3
        samples_size = samples_column * samples_row
        # ./picture_to_test/pic1


        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./inference_tensorboard_dir', graph=sess.graph)
            ckpt = tf.train.get_checkpoint_state(config.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                loader.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                for i in range(1,samples_size+1):#下标从1起
                    img = Image.open('./picture_to_test/pic'+str(i)+'.jpg')
                    # print(img, img.mode, img.size, img.format)
                    # 居中裁剪
                    edge_size = min(img.size[0], img.size[1])
                    if img.size[0] > edge_size:  # 如果x更长，裁剪x
                        start_x = (img.size[0] - edge_size) / 2
                        end_x = start_x + edge_size
                        start_y = 0
                        end_y = edge_size
                    else:
                        start_y = (img.size[0] - edge_size) / 2
                        end_y = start_y + edge_size
                        start_x = 0
                        end_x = edge_size

                    bounds = (start_x, start_y, end_x, end_y)
                    cropped_img = img.crop(bounds)
                    resized_img = cropped_img.resize((48, 48))
                    # 转灰度，转ndarray
                    converted_img = resized_img.convert("L")
                    ndarry_img = np.array(converted_img)
                    # 数值归一
                    ndarry_img = ndarry_img.astype(dtype=np.float32)
                    ndarry_img = ndarry_img * 1. / 255
                    ndarry_img = np.expand_dims(ndarry_img, axis=-1)
                    ndarry_img = np.expand_dims(ndarry_img, axis=0)
                    predict_idx,y_out,summary = sess.run([prediction,y, merged],feed_dict={x:ndarry_img,prob:1.0,bn_training:False})
                    predict_str = prediction_dict[predict_idx[0]]
                    print('pic%d:'%i,end=' ')
                    # print(type(y_out),y_out.shape)
                    for j in range(len(y_out[0])):
                        print(prediction_dict[j],':%.2f'%y_out[0][j],end='\t')
                    print()
                    plt.subplot(samples_row,samples_column,i)
                    plt.title('prediction:'+ predict_str)
                    plt.imshow(img)
                    # print('prediction is ',predict_idx)
                    # if i == 1:#方便打印，注意从1开始
                    writer.add_summary(summary,i)
                plt.show()

            else:
                print('No checkpoint file found')
                return
def main():


    test()


if __name__ == '__main__':
    main()
