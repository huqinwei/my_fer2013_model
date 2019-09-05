#coding:utf-8

#相关配置

#windows环境
#先用fer2013试一下，再改fer2013plus
image_train_path='E:/MachineLearning/datasets/fer2013/train/'
label_train_path='E:/MachineLearning/datasets/fer2013/labels_train.txt'
tfRecord_train='E:/MachineLearning/datasets/fer2013/fer2013_train.tfrecords'

image_valid_path='E:/MachineLearning/datasets/fer2013/valid/'
label_valid_path='E:/MachineLearning/datasets/fer2013/labels_valid.txt'
tfRecord_valid='E:/MachineLearning/datasets/fer2013/fer2013_valid.tfrecords'

image_test_path='E:/MachineLearning/datasets/fer2013/test/'
label_test_path='E:/MachineLearning/datasets/fer2013/labels_test.txt'
tfRecord_test='E:/MachineLearning/datasets/fer2013/fer2013_test.tfrecords'
tfRecord_test_mini='E:/MachineLearning/datasets/fer2013/fer2013_test_mini.tfrecords'#试验用

data_path='E:/MachineLearning/datasets/fer2013/fer2013.csv'
train_data_path='E:/MachineLearning/datasets/fer2013/fer2013_train.csv'
valid_data_path='E:/MachineLearning/datasets/fer2013/fer2013_valid.csv'
test_data_path='E:/MachineLearning/datasets/fer2013/fer2013_test.csv'


data_file='E:/MachineLearning/datasets/fer2013'
#模型存储路径
MODEL_SAVE_PATH="E:/MachineLearning/model_saved/fer2013/"
#模型名称
MODEL_NAME="fer2013_model"
#图片高
img_height=48
#图片宽
img_width=48



# #windows环境
# #先用fer2013试一下，再改fer2013plus
# image_train_path='E:\MachineLearning\datasets\fer2013\train'
# label_train_path='E:\MachineLearning\datasets\fer2013\labels_train.txt'
# tfRecord_train='E:\MachineLearning\datasets\fer2013\fer2013_train.tfrecords'
#
# image_valid_path='E:\MachineLearning\datasets\fer2013\valid/'
# label_valid_path='E:\MachineLearning\datasets\fer2013\labels_valid.txt'
# tfRecord_valid='E:\MachineLearning\datasets\fer2013\fer2013_valid.tfrecords'
#
# image_test_path='E:\MachineLearning\datasets\fer2013\test/'
# label_test_path='E:\MachineLearning\datasets\fer2013\labels_test.txt'
# tfRecord_test='E:\MachineLearning\datasets\fer2013\fer2013_test.tfrecords'
#
# data_path='E:\MachineLearning\datasets\fer2013\fer2013.csv'
# train_data_path='E:\MachineLearning\datasets\fer2013\fer2013_train.csv'
# valid_data_path='E:\MachineLearning\datasets\fer2013\fer2013_valid.csv'
# test_data_path='E:\MachineLearning\datasets\fer2013\fer2013_test.csv'
#
#
# data_file='E:\MachineLearning\datasets\fer2013'
# #模型存储路径
# MODEL_SAVE_PATH="E:\MachineLearning\model_saved\fer2013"
# #模型名称
# MODEL_NAME="fer2013_model"
# #图片高
# img_height=48
# #图片宽
# img_width=48










#linux环境


# #先用fer2013试一下，再改fer2013plus
# image_train_path='/home/qw/Documents/datasets/fer2013/train/'
# label_train_path='/home/qw/Documents/datasets/fer2013/labels_train.txt'
# tfRecord_train='/home/qw/Documents/datasets/fer2013/fer2013_train.tfrecords'
#
# image_valid_path='/home/qw/Documents/datasets/fer2013/valid/'
# label_valid_path='/home/qw/Documents/datasets/fer2013/labels_valid.txt'
# tfRecord_valid='/home/qw/Documents/datasets/fer2013/fer2013_valid.tfrecords'
#
# image_test_path='/home/qw/Documents/datasets/fer2013/test/'
# label_test_path='/home/qw/Documents/datasets/fer2013/labels_test.txt'
# tfRecord_test='/home/qw/Documents/datasets/fer2013/fer2013_test.tfrecords'
#
# data_path='/home/qw/Documents/datasets/fer2013/fer2013.csv'
# train_data_path='/home/qw/Documents/datasets/fer2013/fer2013_train.csv'
# valid_data_path='/home/qw/Documents/datasets/fer2013/fer2013_valid.csv'
# test_data_path='/home/qw/Documents/datasets/fer2013/fer2013_test.csv'
#
#
# data_file='/home/qw/Documents/datasets/fer2013'
# #模型存储路径
# MODEL_SAVE_PATH="/home/qw/Documents/model_saved/fer2013"
# #模型名称
# MODEL_NAME="fer2013_model"
# #图片高
# img_height=48
# #图片宽
# img_width=48


#plus数据，其实没有数据，只是label，数据还得用旧的！标签是10个，不是7个。
#/home/qw/Documents/datasets/FERPlus
#
# image_train_path='/home/qw/Documents/datasets/FERPlus/train/'
# label_train_path='/home/qw/Documents/datasets/FERPlus/labels_train.txt'
# tfRecord_train='/home/qw/Documents/datasets/FERPlus/fer2013_train.tfrecords'
#
# image_valid_path='/home/qw/Documents/datasets/FERPlus/valid/'
# label_valid_path='/home/qw/Documents/datasets/FERPlus/labels_valid.txt'
# tfRecord_valid='/home/qw/Documents/datasets/FERPlus/fer2013_valid.tfrecords'
#
# image_test_path='/home/qw/Documents/datasets/FERPlus/test/'
# label_test_path='/home/qw/Documents/datasets/FERPlus/labels_test.txt'
# tfRecord_test='/home/qw/Documents/datasets/FERPlus/fer2013_test.tfrecords'
#
# data_path='/home/qw/Documents/datasets/FERPlus/fer2013.csv'
# train_data_path='/home/qw/Documents/datasets/FERPlus/fer2013_train.csv'
# valid_data_path='/home/qw/Documents/datasets/FERPlus/fer2013_valid.csv'
# test_data_path='/home/qw/Documents/datasets/FERPlus/fer2013_test.csv'
#
#
# data_file='/home/qw/Documents/datasets/FERPlus'
# #模型存储路径
# MODEL_SAVE_PATH="/home/qw/Documents/model_saved/fer2013plus"
# #模型名称
# MODEL_NAME="fer2013plus_model"
# #图片高
# img_height=48
# #图片宽
# img_width=48


