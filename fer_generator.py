#coding:utf-8
#提供一些数据处理的接口并进行处理，包括原始数据转tfrecord，从tfrecord获取数据队列来训练网络。
import tensorflow as tf
from PIL import Image
import os
import random
import pandas as pd
import numpy as np
import fer_config as config

#将训练集和测试集分开
def split_data():
    data = pd.read_csv(config.data_path)
    images = data["pixels"]
    labels = data["emotion"]
    usages = data["Usage"]
    img_train=[]
    img_valid = []
    img_test=[]
    label_train=[]
    label_valid = []
    label_test=[]
    for i in range(len(images)):
        # 训练集
        if usages[i]=="Training":
            img_train.append(images[i])
            label_train.append(labels[i])
        # 验证集
        elif usages[i]=="PrivateTest":
            img_valid.append(images[i])
            label_valid.append(labels[i])
        # 测试集
        else:
            img_test.append(images[i])
            label_test.append(labels[i])
    data_train=pd.DataFrame(data={"pixels":img_train,"emotion":label_train},
                           columns=["pixels","emotion"])
    data_valid = pd.DataFrame(data={"pixels": img_valid, "emotion": label_valid},
                              columns=["pixels", "emotion"])
    data_test=pd.DataFrame(data={"pixels":img_test,"emotion":label_test},
                           columns=["pixels","emotion"])
    # 存入CSV文件
    data_train.to_csv(config.train_data_path, encoding="utf_8_sig", index=False)
    data_valid.to_csv(config.valid_data_path, encoding="utf_8_sig", index=False)
    data_test.to_csv(config.test_data_path, encoding="utf_8_sig", index=False)

# 生成图片和label文件
def generate_images_and_labels():
    images_train, labels_train=read_data(config.train_data_path)
    images_valid, labels_valid = read_data(config.valid_data_path)
    images_test, labels_test = read_data(config.test_data_path)

        
    def generate(images,labels,images_path,lables_path):
        #补充一个路径建立
        images_path_is_exist = os.path.exists(images_path)
        if not images_path_is_exist:
            os.makedirs(images_path)
            print('create ',images_path)
        with open(lables_path, 'w', encoding="utf-8") as f:
            for i in range(len(images)):
                # 转为灰度图
                image = Image.fromarray(np.reshape(images[i], (config.img_width, config.img_height))).convert('L')
                img_path = str(i) + "_" + str(labels[i]) + '.png'
                # 存储图片
                image.save(images_path + img_path)
                # 写入label文件
                f.write(img_path + " " + str(labels[i]))
                f.write("\n")
                print('the number of picture saved:', i)
        f.close()


    generate(images_train,labels_train,config.image_train_path,config.label_train_path)
    print("end generating train_images")

    generate(images_valid, labels_valid, config.image_valid_path, config.label_valid_path)
    print("end generating valid_images")

    generate(images_test, labels_test, config.image_test_path, config.label_test_path)
    print("end generating test_images")

# 读取CSV文件中数据
def read_data(path):
    data = pd.read_csv(path)
    images=[np.array(p.split(" "),dtype=int) for p in data["pixels"]]
    labels=data["emotion"]
    return images,labels

# 写入tfRecord文件
def write_tfRecord(tfRecordName, image_path, label_path):
    # 得到一个TFRecordWriter
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()
    # 打乱数据
    random.shuffle(contents)
    f.close()
    for content in contents:
    # for content in contents[:2]:
        value = content.split()
        img_path = image_path + value[0]
        # print('img_path is ',img_path)#为了测试时查找对应文件，因为index都没了。。。
        img = Image.open(img_path).convert("L")
        img_raw = img.tobytes()
        labels = [0] * 7
        labels[int(value[1])] = 1

        example = tf.train.Example(
            features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
            }))
        writer.write(example.SerializeToString())
        num_pic += 1
        print('the number of picture:', num_pic)
    writer.close()
    print('write tfrecord successful')
# 生成tfRecord文件
def generate_tfRecord():
    isExists = os.path.exists(config.data_file)
    if not isExists:
        os.makedirs(config.data_file)
        print('The directory was created successfully')
    else:
        print('directory already exists')
    write_tfRecord(config.tfRecord_train, config.image_train_path, config.label_train_path)
    write_tfRecord(config.tfRecord_valid, config.image_valid_path, config.label_valid_path)
    write_tfRecord(config.tfRecord_test, config.image_test_path, config.label_test_path)
# 读取tfRecord文件
def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    print('filename_queue:',filename_queue)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label':tf.FixedLenFeature([7],tf.int64),
                                           'img_raw':tf.FixedLenFeature([],tf.string)
                                                 })
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img.set_shape([config.img_height * config.img_height])

    img = tf.cast(img,tf.float32)*(1./255)
    label = tf.cast(features['label'],tf.float32)
    return img,label

# 读取tfRecord文件，数据增强版。带随机的左右翻转。
def read_tfRecord_DA(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    print('filename_queue:',filename_queue)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label':tf.FixedLenFeature([7],tf.int64),
                                           'img_raw':tf.FixedLenFeature([],tf.string)
                                                 })
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    print('img type is ',type(img))
    print('img shape is ',img.shape)

    #todo 加随机性
    # 此时是tensor，需要使用tf的接口来实现翻转,image：形状为[batch, height, width, channels]的4-D张量或形状为[height, width, channels]的3-D张量.

    flipped = tf.image.flip_left_right(img)#直接能翻吗？还有一个维度呢！此时还没有，多加的维度是反向传播时候做的。

    img.set_shape([config.img_height * config.img_height])

    img = tf.cast(img,tf.float32)*(1./255)
    label = tf.cast(features['label'],tf.float32)
    return img,label


# 批量读取数据
def get_tfrecord(num, tfRecord_path,data_augment = False):
    if data_augment:
        print('true')
        img, label = read_tfRecord_DA(tfRecord_path)
    else:
        print('false')
        img,label=read_tfRecord(tfRecord_path)
    img_batch,label_batch = tf.train.shuffle_batch([img,label],
                                                   batch_size =num,
                                                   num_threads=2,
                                                   capacity=10000,
                                                   min_after_dequeue=5000)
    return img_batch,label_batch

#def get_tfrecord0(num, tfRecord_path):#read_tfRecord挡着，没法直接sess.run()让地址为空报个错
#    img,label=read_tfRecord(tfRecord_path)
#    return img,label

def main():
    #根据需要运行，完整的就是三步，如果都有了，可能三步都不运行。
    # split_data()#分解成三个数据集并存储。
    # generate_images_and_labels()
    # generate_tfRecord()


    write_tfRecord(config.tfRecord_test_mini, config.image_test_path, config.label_test_path)#专门生成一个mini集合
    pass

if __name__ == '__main__':
    main()






