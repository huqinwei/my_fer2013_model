#用mnist，处理一下原来的数据，把某一类减少到十分之一，然后看这一类的准确率和其他类的准确率有何不同，然后给这一类的loss来个十倍权重

#思路：数据label带着分类信息，预处理，或者内部argmax处理，让每个batch内的不同数据能携带不同分类信息，就能决定是否乘以系数
#问题是tf.cond是整个流的吧，不能给单个数据。tf.where可以针对单个数据。当然，还有切片方法，现在直接用切片就可实现。

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets(train_dir = 'mnist_data',one_hot = True)

train_images = mnist.train.images#(55000, 784)
train_labels = mnist.train.labels#(55000, 10)
if 1:
    # 5444,6179,5470,5638,5307,4987,5417,5715,5389,5454
    n_classes = 10
    specified_class_idx = 3#指定一个类，数据缩水
    delete_nums = 5000#删5000个，还剩下638
    del_idx_list = []#记录下来满足条件的下标，然后一次性删除
    for i in range(train_images.shape[0]):
        if np.argmax(train_labels[i]) == specified_class_idx:
            if delete_nums == 0:
                break
            del_idx_list.append(i)
            delete_nums -= 1
    print('del_idx_list:',del_idx_list)
    new_train_images = np.delete(train_images,del_idx_list,axis=0)
    new_train_labels = np.delete(train_labels,del_idx_list,axis=0)
    print('new_train_images.shape:',new_train_images.shape)
    print('new_train_labels.shape:',new_train_labels.shape)
    # np.random.shuffle(new_train_images)#这个shuffle不能用，不对应了，先不洗牌了，影响不大（可用随机index同步处理）
    # np.random.shuffle(new_train_labels)
else:#保持原样数据，对比
    new_train_images = train_images
    new_train_labels = train_labels

test_size = mnist.test.images.shape[0]
print('test_size:',test_size)
specified_test_images = []
specified_test_labels = []
other_test_images = []
other_test_labels = []
for i in range(test_size):
    if np.argmax(mnist.test.labels[i]) == specified_class_idx:
        specified_test_images.append(mnist.test.images[i])
        specified_test_labels.append(mnist.test.labels[i])
    else:
        other_test_images.append(mnist.test.images[i])
        other_test_labels.append(mnist.test.labels[i])

ndarray_specified_test_images = np.array(specified_test_images)
ndarray_specified_test_labels = np.array(specified_test_labels)
ndarray_other_test_images = np.array(other_test_images)
ndarray_other_test_labels = np.array(other_test_labels)
print('ndarray_specified_test_labels:',ndarray_specified_test_labels)

learning_rate = 0.0005
training_epochs = 200
batch_size = 256
display_step = 20

x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y_ = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
#如果w和b初始化完全用zeros：训练前9%，训练完11%
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.ones([10])/10.)
tf.summary.histogram(name='W',values=W)
tf.summary.histogram(name='b',values=b)

#暂时只用一层，两层的话，可能是relu影响，手动的cross_entropy都是Nan了。tf接口为什么能跑，奇怪。
y_without_softmax = tf.matmul(x, W) + b
tf.summary.histogram(name='y_without_softmax',values=y_without_softmax)
y = tf.nn.softmax(y_without_softmax) # Softmax
tf.summary.histogram(name='y',values=y)

# Minimize error using cross entropy
specified_class_weight=10 # 指定类
other_class_weight=1 # 其他类

# #切片，会出现Nan值和infinity，tf.clip_by_value没解决,是RELU激活的问题，网络只留一层，不加激活，也不能用clip，也会影响训练效果的
#注意顺序，外部mean，内部sum，log内是prediction
cross_entropy = tf.reduce_mean(
    -specified_class_weight*tf.reduce_sum((y_[:,specified_class_idx] * tf.log(y[:,specified_class_idx])))
    -other_class_weight*tf.reduce_sum((y_[:,:specified_class_idx] * tf.log(y[:,:specified_class_idx])))
    -other_class_weight*tf.reduce_sum((y_[:,specified_class_idx+1:] * tf.log(y[:,specified_class_idx+1:])))
)
#tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
#如果reduce_sum没有reduction_indices=1，外部的mean还需要？

# 这个也是外部用reduce_mean处理样本间的关系，所以用tf.where可能能实现权重吗？可能吧，先不试了，比较麻烦。
# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_without_softmax,labels=tf.argmax(y_,axis=1)))

cost = cross_entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
# Test model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Calculate accuracy for 3000 examples
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy:',accuracy)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('mnist_unbalanced_sample_log')
writer_test = tf.summary.FileWriter('mnist_unbalanced_sample_log_test')
writer_spec = tf.summary.FileWriter('mnist_unbalanced_sample_log_specified')
writer_other = tf.summary.FileWriter('mnist_unbalanced_sample_log_other')
# Start training
with tf.Session() as sess:
    sess.run(init)
    print("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y_: mnist.test.labels[:3000]}))

    # Training cycle
    n_samples = new_train_images.shape[0]
    batch_nums = n_samples//batch_size
    print('batch_nums:',batch_nums)
    for i in range(training_epochs * batch_nums):#新的步数，为了下边的循环好写。
        # avg_cost = 0.
        start = i*batch_size % n_samples
        end = (i+1)*batch_size % n_samples
        if end < start:#不要这种。
            print('continue!')
            continue
        batch_xs = new_train_images[start: end]
        batch_ys = new_train_labels[start: end]

        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                      y_: batch_ys})

        if (i+1) % display_step == 0:#只是batch的cost，也不算太好。
            # accu,y_val,y_label = sess.run([accuracy,y,y_],feed_dict={x: mnist.train.images[:3000], y_: mnist.train.labels[:3000]})
            # print("train Accuracy:", accu)
            # print('y_val:',y_val)
            # print('y_label:',y_label)

            accu,summary = sess.run([accuracy,merged],feed_dict={x: mnist.train.images[:3000], y_: mnist.train.labels[:3000]})
            print("train Accuracy:", accu)
            writer.add_summary(summary,i)

            accu,summary = sess.run([accuracy,merged],feed_dict={x: mnist.test.images[:3000], y_: mnist.test.labels[:3000]})
            print("test Accuracy:", accu)
            writer_test.add_summary(summary, i)

            accu,summary = sess.run([accuracy,merged],feed_dict={x: ndarray_specified_test_images, y_: ndarray_specified_test_labels})
            print("spec Accuracy:", accu)
            writer_spec.add_summary(summary, i)

            accu,summary = sess.run([accuracy,merged],feed_dict={x: ndarray_other_test_images, y_: ndarray_other_test_labels})
            print("other Accuracy:", accu)
            writer_other.add_summary(summary, i)

    print("Optimization Finished!")

