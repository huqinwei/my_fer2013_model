import tensorflow as tf
import fer_forward
import os
import fer_generator
import numpy as np
import time
import fer_config as config

BATCH_SIZE = 256#128
LEARNING_RATE_BASE = 0.0005
LEARNING_RATE_DECAY = 0.99
STEPS = 100000
MOVING_AVERAGE_DECAY = 0.5
train_num_examples=28709
#定义反向传播
def backward():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 输入图像size是可变的，通过配置文件，lenet输入通道数是1，灰度图，这个应该得先处理了，原图可能不是灰度的
    x = tf.placeholder(tf.float32, [BATCH_SIZE, config.img_width,
                                    config.img_height, fer_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, fer_forward.OUTPUT_NODE])
    #建立网络和损失函数
    y = fer_forward.forward(x,True)
    #损失
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('regularization_losses'))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 用了EMA，测试的时候取EMA，训练的时候反馈正确率用的是weights
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#BN需要依赖操作
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    #准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    saver = tf.train.Saver()
    # 批量获取数据
    img_batch,label_batch=fer_generator.get_tfrecord(BATCH_SIZE,config.tfRecord_train)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()#队列报错，需要这句？没解决
        # 继续训练
        ckpt=tf.train.get_checkpoint_state(config.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)

        # 创建一个线程协调器
        coord=tf.train.Coordinator()
        # 启动入队线程
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        #img_batch,label_batch=fer_generator.get_tfrecord(BATCH_SIZE,config.tfRecord_train)##这个位置是血的教训
        for i in range(STEPS):
            xs, ys = sess.run([img_batch,label_batch])

            reshape_xs = np.reshape(xs, (BATCH_SIZE,
                                         config.img_width,
                                         config.img_height,
                                         fer_forward.NUM_CHANNELS))
            # 训练更新
            _, loss_value, accuracy_value, step,lr = sess.run([train_op, loss, accuracy, global_step, learning_rate],
                                                           feed_dict={x: reshape_xs, y_: ys})
            if (i + 1) % 100 == 0:
                print("%s : After %d training step(s),lr is %g loss,accuracy on training batch is %g , %g." % (
                time.strftime('%Y-%m-%d %H:%M:%S'), step, lr,loss_value,accuracy_value))

                saver.save(sess, os.path.join(config.MODEL_SAVE_PATH, config.MODEL_NAME), global_step=global_step)

        coord.request_stop()
        coord.join(threads)

def main():
    backward()

if __name__ == '__main__':
    main()


