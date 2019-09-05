#测一下把隐藏参数获取并且打印到tensorboard
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

#将x1训练拟合label1，然后拿x2和x3做测试，预期的输出其实是x2和x3经过转换，和x1的转换结果一致。
sess = tf.Session()
x1_ = np.array([[1., 2., 3.], [11., 12., 13.]], dtype=np.float32)  #
label1_ = tf.constant([[2., 4., 6.], [22., 24., 26.]], dtype=tf.float32)
x2_ = np.array([[3., 3., 3.], [10., 14., 20.]], dtype=np.float32)
x3_ = np.array([[4., 2., 3.], [5., 3., 5.]], dtype=np.float32)  # 和x2输出相近，作为对照
input_ = tf.placeholder(shape=[2, 3], dtype=tf.float32)
y1 = tf.layers.batch_normalization(input_, training=True, name="bn_1", epsilon=1e-09)
loss = tf.reduce_sum(tf.square(y1 - label1_))

#get到bn层的参数去记录。
gamma = sess.graph.get_tensor_by_name(name = 'bn_1/gamma:0')
beta = sess.graph.get_tensor_by_name(name = 'bn_1/beta:0')
moving_mean = sess.graph.get_tensor_by_name(name = 'bn_1/moving_mean:0')
moving_variance = sess.graph.get_tensor_by_name(name = 'bn_1/moving_variance:0')

tf.summary.histogram('bn/gamma',gamma)
tf.summary.histogram('bn/beta',beta)
tf.summary.histogram('bn/moving_mean',moving_mean)
tf.summary.histogram('bn/moving_variance',moving_variance)

tf.summary.histogram('input',input_)
tf.summary.histogram('label',label1_)
tf.summary.histogram('output',y1)
tf.summary.scalar('loss',loss)

merged = tf.summary.merge_all()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print('update_ops:', update_ops)
with tf.control_dependencies(update_ops):  # 可否省略主动关联更新操作？
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init = tf.global_variables_initializer()

sess.run(init)
sess.run(gamma)
writer = tf.summary.FileWriter(logdir='bn_tensorboard_demo', graph=sess.graph)
test_writer = tf.summary.FileWriter(logdir='bn_tensorboard_demo_test', graph=sess.graph)
for i in range(800):
    if i % 20 == 0:
        output,summary = sess.run([y1,merged], feed_dict={input_: x1_})
        print('after %d steps, train output(y) is :'% (i),output)
        writer.add_summary(summary,i)

        output,summary = sess.run([y1,merged], feed_dict={input_: x1_})
        print('after %d steps, test output(y) is :'% (i),output)
        test_writer.add_summary(summary,i)

    sess.run(train_op, feed_dict={input_: x1_})
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!after train!!!!!!!!!!!!!!!!!!!!!!!!!')
# 用训练模式预测的会符合预期一些！！
print('x1  y1:', sess.run(y1, feed_dict={input_: x1_}))
print('x2  y1:', sess.run(y1, feed_dict={input_: x2_}))
print('x3  y1:', sess.run(y1, feed_dict={input_: x3_}))
# 用预测模式预测的会偏差一些！！因为预测模式使用了moving_mean和moving_variance，这不是根据数据自身量身定制的！！！！

print(tf.global_variables())

############################################
#下边是读取部分。
#
# #手动variables的BN，加载model，使用非训练模式
# #其实如果你不保存变量，会报错的。。。其实也不用担心这里忘了保存。
# import tensorflow as tf
# import tensorflow as tf
# import numpy as np
# from tensorflow.python.training import moving_averages
# x1_ = np.array([[1.,2.,3.],[11.,12.,13.]], dtype=np.float32)#
# label1_ = tf.constant([[2.,4.,6.],[22.,24.,26.]],dtype=tf.float32)
#
# x2_ = np.array([[3.,3.,3.],[10.,14.,20.]],dtype=np.float32)
# x3_ = np.array([[4.,2.,3.],[5.,3.,5.]],dtype=np.float32)#和x2输出相近，作为对照
# x4_ = np.array([[6.,2.,2.],[21.,26.,31.]],dtype=np.float32)#和x2输出相近，作为对照
#
# input_ = tf.placeholder(shape=[2,3],dtype = tf.float32)
#
#
# #默认是False，如果预测时保持和训练相同的training=True，就不对了
# prediction = tf.layers.batch_normalization(input_, training=False,name="bn_1",epsilon=1e-09)#如果这里用默认或者True呢
# # prediction = tf.layers.batch_normalization(input_, name="bn_1",epsilon=1e-09)#如果这里用默认或者True呢
# # prediction = tf.layers.batch_normalization(input_, training=True,name="bn_1",epsilon=1e-09)#如果这里用默认或者True呢
#
# loader = tf.train.Saver()
# with tf.Session() as sess:
#
#     loader.restore(sess, './BN_variables_save2.ckpt')
#     print(tf.global_variables())
#     print('x1-prediction',sess.run(prediction,feed_dict={input_:x1_}))
#     print('x2-prediction',sess.run(prediction,feed_dict={input_:x2_}))
#     print('x3-prediction',sess.run(prediction,feed_dict={input_:x3_}))
#     print('x4-prediction',sess.run(prediction,feed_dict={input_:x4_}))
#
# #与”预期“保持一致
# # x1-prediction [[  2.0474062    4.06059265   6.07377148]
# #  [ 22.11075783  24.12394524  26.13712502]]
# # x2-prediction [[  6.06007671   6.06692791   6.07377148]
# #  [ 20.10442352  28.13661575  40.18147278]]
# # x3-prediction [[  8.06641197   4.06059265   6.07377148]
# #  [ 10.07274723   6.06692791  10.08644199]]
# #如果仍然是训练阶段，则所有数据都接近label的2\4\6\22\24\26
# # x1-prediction [[  1.99999964   3.99999928   6.        ]
# #  [ 21.99997711  23.99997711  25.99997711]]
# # x2-prediction [[  1.99999905   3.99999809   6.        ]
# #  [ 21.99997711  23.99997711  25.99997711]]
# # x3-prediction [[  2.           4.           6.        ]
# #  [ 21.99997711  23.99997711  25.99997711]]

