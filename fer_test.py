# coding:utf-8
import time
import tensorflow as tf
import fer_forward
import backward
import fer_generator
import numpy as np
import fer_config as config
import os

TEST_INTERVAL_SECS = 5
TOTAL_TEST_NUM = 3589
MINI_BATCH = 100#分批测试的话，注意多线程问题。保证是多线程，才能每次确实shuffle拿出了不一样的数据。所以线程在外，循环在内

#3589，test下标从0~3588，valid同样是0~3588
#这里边没用valid数据集，严格讲，以valid数据集为准，向test数据集泛化。
#需要使用valid和test同时处理过拟合问题的话，手动改一下数据文件来源。

#更多可以选的操作，把train和valid放一起，交叉验证。

#为了不抢GPU，不OOM，限制只用CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'



def test():
    # 实例化一个数据流图并作为整个 tensorflow 运行环境的默认图
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [MINI_BATCH, config.img_width,
                                    config.img_height, fer_forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, fer_forward.OUTPUT_NODE])

        prob = tf.placeholder(tf.float32)
        bn_training = tf.placeholder(tf.bool)
        # y = fer_forward.forward(x, keep_prob=prob)
        y,dict_ret = fer_forward.forward(x,keep_prob=prob,bn_enable=True,bn_training=bn_training)
        
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()#生成ema替代原变量的映射关系。
        

        loader = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 批量获取测试数据
        img_batch, label_batch = fer_generator.get_tfrecord(MINI_BATCH, config.tfRecord_test)
        for i in range(3):
            # 创建一个会话
            with tf.Session() as sess:
                # 通过checkpoint文件找到模型文件名
                ckpt = tf.train.get_checkpoint_state(config.MODEL_SAVE_PATH)
                # 如果模型存在
                if ckpt and ckpt.model_checkpoint_path:

                    loader.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    # 创建一个线程协调器
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                    
                    iterations = int(TOTAL_TEST_NUM / MINI_BATCH)
                    total_accuracy_score = 0
                    for i in range(iterations):
                        xs, ys = sess.run([img_batch, label_batch])#一定要把这步扔到循环内部。
                        # reshape测试输入数据xs
                        reshape_xs = np.reshape(xs, (MINI_BATCH,
                                                     config.img_width,
                                                     config.img_height,
                                                     fer_forward.NUM_CHANNELS))

                        accuracy_score = sess.run(accuracy, feed_dict={x: reshape_xs, y_: ys, prob:1.0, bn_training:False})

                        print("%g" % (accuracy_score),end=', ')
                        total_accuracy_score += accuracy_score
                        

                    # 输出global_step和准确率
                    print("After %s training step(s), test accuracy = %g" % (global_step, total_accuracy_score / iterations))
                    # 终止所有线程
                    coord.request_stop()
                    coord.join(threads)

                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    test()


if __name__ == '__main__':
    main()
