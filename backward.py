import tensorflow as tf
import fer_forward
import os
import fer_generator
import numpy as np
import time
import fer_config as config
import shutil

TOTAL_TEST_NUM = 3589
MINI_BATCH = 256

BATCH_SIZE = 256#128
#如果用decay还不加保底学习率，观察一下最后学习率是否太低，可以中断训练，改基础学习率，算上decay，也不怕学习率高。（不一定需要，毕竟已经过拟合了）
LEARNING_RATE_BASE = 0.005#临时做的中途调整，因为已经243359，有衰减，，，lr is 1.49615e-13，，，，默认是0.0005
LEARNING_RATE_DECAY = 0.99
STEPS = 200000
MOVING_AVERAGE_DECAY = 0.5
train_num_examples=28709

def cross_entropy(y,y_):
    # 原损失
    # ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # cem = tf.reduce_mean(ce)

    # [3995, 436, 4097, 7215, 4830, 3171, 4965]
    specified_class_idx = 1

    # 指定类#如果这个是10倍，是不是会稀释正则化的系数？折中？5和0.5？，这个类占总数的65分之一，乘以10的话，影响大不大？测完了便知。
    specified_class_weight = 10
    other_class_weight = 1  # 其他类
    # 手动的话，需要softmax？还是改用sigmoid？
    # softmax会导致Nan？
    tf.summary.histogram(name='y/before_softmax',values=y)
    y = tf.nn.softmax(y)
    tf.summary.histogram(name='y/after_softmax',values=y)
    # tf.summary.histogram(name='y/before_sigmoid', values=y)
    # y = tf.nn.sigmoid(y)
    # tf.summary.histogram(name='y/after_sigmoid', values=y)
    print('!!!!!!!!!!!!!!!y is ', y)
    # 这个切的方法对不对？总感觉不对？连argmax都没有？不成了切列了？但是好像除了这一列，也确实都是0.
    # reduce_sum没加参数reduction_indices=1，所以直接就成scalar了
    # tf.clip_by_value(,1e-10,1.0)  loss = tf.log(tf.clip_by_value(y,1e-8,tf.reduce_max(y)))#保留最大值，多了很多细节
    ce1_yhat = y_[:, specified_class_idx]
    tf.summary.histogram(name='ce/ce1_yhat', values=ce1_yhat)
    ce1_y_before_clip = y[:, specified_class_idx]
    tf.summary.histogram(name='ce/ce1_y_before_clip', values=ce1_y_before_clip)
    ce1_y_after_clip = tf.clip_by_value(ce1_y_before_clip, 1e-10, tf.reduce_max(ce1_y_before_clip))
    tf.summary.histogram(name='ce/ce1_y_after_clip', values=ce1_y_after_clip)
    ce1_y_after_log = tf.log(ce1_y_after_clip)
    tf.summary.histogram(name='ce/ce1_y_after_log', values=ce1_y_after_log)

    ce2_yhat = y_[:, :specified_class_idx]
    tf.summary.histogram(name='ce/ce2_yhat', values=ce2_yhat)
    ce2_y_before_clip = y[:, :specified_class_idx]
    tf.summary.histogram(name='ce/ce2_y_before_clip', values=ce2_y_before_clip)
    ce2_y_after_clip = tf.clip_by_value(ce2_y_before_clip, 1e-10, tf.reduce_max(ce2_y_before_clip))
    tf.summary.histogram(name='ce/ce2_y_after_clip', values=ce2_y_after_clip)
    ce2_y_after_log = tf.log(ce2_y_after_clip)
    tf.summary.histogram(name='ce/ce2_y_after_log', values=ce2_y_after_log)

    # todo，拼接起来合并，不过concat暂时不成功。说是(?)和(?,5)不匹配
    ce3_yhat = y_[:, specified_class_idx + 1:]
    tf.summary.histogram(name='ce/ce3_yhat', values=ce3_yhat)
    ce3_y_before_clip = y[:, specified_class_idx + 1:]
    tf.summary.histogram(name='ce/ce3_y_before_clip', values=ce3_y_before_clip)
    ce3_y_after_clip = tf.clip_by_value(ce3_y_before_clip, 1e-10, tf.reduce_max(ce3_y_before_clip))
    tf.summary.histogram(name='ce/ce3_y_after_clip', values=ce3_y_after_clip)
    ce3_y_after_log = tf.log(ce3_y_after_clip)
    tf.summary.histogram(name='ce/ce3_y_after_log', values=ce3_y_after_log)

    ce1 = -specified_class_weight * tf.reduce_sum(ce1_yhat * ce1_y_after_log)
    ce2 = -specified_class_weight * tf.reduce_sum(ce2_yhat * ce2_y_after_log)
    ce3 = -specified_class_weight * tf.reduce_sum(ce3_yhat * ce3_y_after_log)
    cem = tf.reduce_mean(ce1 + ce2 + ce3)
    tf.summary.scalar(name='ce/ce1', tensor=ce1)  # histogram
    tf.summary.scalar(name='ce/ce2', tensor=ce2)
    tf.summary.scalar(name='ce/ce3', tensor=ce3)
    tf.summary.scalar(name='ce/cem', tensor=cem)


    print('ce1_yhat:',ce1_yhat)
    print('ce2_yhat:',ce2_yhat)
    print('ce3_yhat:',ce3_yhat)

    print('ce1_y_before_clip:',ce1_y_before_clip)
    print('ce2_y_before_clip:',ce2_y_before_clip)
    print('ce3_y_before_clip:',ce3_y_before_clip)


    return cem

def tensorboard_print(return_dict):
    print('return_dict:',return_dict)
    #return_dict['relu1_output']，shape = (256, 48, 48, 64),成功的是这种：(256, 48, 48, 1)
    #错误写法：return_dict['relu1_output'][:][:][:][1]，其实是多次提取全部，最后提取第[1]个样本,正确写法：return_dict['relu1_output'][:,:,:,1]

    tf.summary.image('relu1_channel1',tf.expand_dims(input=return_dict['relu1_output'][:,:,:,1],axis=-1))#记录输出
    tf.summary.image('relu1_channel2',tf.expand_dims(input=return_dict['relu1_output'][:,:,:,2],axis=-1))#记录输出
    tf.summary.image('relu1_channel3',tf.expand_dims(input=return_dict['relu1_output'][:,:,:,3],axis=-1))#记录输出
    tf.summary.image('relu2_channel1',tf.expand_dims(input=return_dict['relu2_output'][:,:,:,1],axis=-1))#记录输出
    tf.summary.image('relu2_channel2',tf.expand_dims(input=return_dict['relu2_output'][:,:,:,2],axis=-1))#记录输出
    tf.summary.image('relu2_channel3',tf.expand_dims(input=return_dict['relu2_output'][:,:,:,3],axis=-1))#记录输出
    tf.summary.image('relu3_channel1',tf.expand_dims(input=return_dict['relu3_output'][:,:,:,1],axis=-1))#记录输出
    tf.summary.image('relu3_channel2',tf.expand_dims(input=return_dict['relu3_output'][:,:,:,2],axis=-1))#记录输出
    tf.summary.image('relu3_channel3',tf.expand_dims(input=return_dict['relu3_output'][:,:,:,3],axis=-1))#记录输出
    print(return_dict['relu3_output'])#256*12*12*128
    print(return_dict['pool3_output'])#shape=(256, 6, 6, 128),256*6*6*128

    #全连接也不是图形,不能记录

#定义反向传播
def backward():
    flip_control = tf.random_uniform(shape=[BATCH_SIZE])#控制翻转，都从[1]改成[batch_size]，tf.cond换成tf.where。
    rotate_control = tf.random_uniform(shape=[BATCH_SIZE])
    rotate_angle = tf.random_uniform(shape=[1],minval=-0.5,maxval=0.5,dtype=tf.float32)#我自己测0.5还可接受，因为有些图本来就偏，不能偏太大。
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 输入图像size是可变的，通过配置文件，lenet输入通道数是1，灰度图，这个应该得先处理了，原图可能不是灰度的
    x = tf.placeholder(tf.float32, [BATCH_SIZE, config.img_width,
                                    config.img_height, fer_forward.NUM_CHANNELS],name='x')
    y_ = tf.placeholder(tf.float32, [None, fer_forward.OUTPUT_NODE],name='y_')
    prob = tf.placeholder(tf.float32,name='keep_prob')
    bn_training = tf.placeholder(tf.bool,name='bn_training')
    is_data_augment = tf.placeholder(tf.bool,name='data_augment')#是否数据增强，测准确度的时候都不应该带。#这个其实在提取数据的计算图，不在网络的计算图。。。。
    #建立网络和损失函数
    y,return_dict = fer_forward.forward(x,keep_prob=prob,bn_enable=True,bn_training=bn_training)
    cem = cross_entropy(y,y_)

    loss = cem + tf.add_n(tf.get_collection('regularization_losses'))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 用了EMA，测试的时候取EMA，训练的时候反馈正确率用的是weights
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    tensorboard_print(return_dict)

    print('ema:',ema)
    print('tf.trainable_variables():',tf.trainable_variables())
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('update_ops:', update_ops)
    print('tf.GraphKeys.UPDATE_OPS:',tf.GraphKeys.UPDATE_OPS)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#BN需要依赖操作
    with tf.control_dependencies(update_ops):
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')
    #准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    summary_cem = tf.summary.scalar('cem', cem)
    summary_loss = tf.summary.scalar('loss', loss)
    summary_accuracy = tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()
    # 批量获取数据
    img_batch,label_batch=fer_generator.get_tfrecord(BATCH_SIZE,config.tfRecord_train)
    reshaped_img_batch = tf.reshape(img_batch, shape=[BATCH_SIZE, config.img_width, config.img_height,
                                                      fer_forward.NUM_CHANNELS])
    print('reshaped_img_batch:',reshaped_img_batch)
    tf.summary.image('img_input',reshaped_img_batch)#记录输入图片

    #数据增强应该可选
    # flipped_img_batch = tf.cond(is_data_augment,
    #                             lambda:tf.where(flip_control >= 0.5,
    #                                      lambda:tf.image.flip_left_right(reshaped_img_batch),
    #                                      lambda:reshaped_img_batch),
    #                             lambda:reshaped_img_batch)
    # #                             lambda: reshaped_img_batch)
    # rotated_img_batch = tf.cond(is_data_augment,
    #                             tf.where(rotate_control >= 0.5,
    #                                      tf.contrib.image.rotate(flipped_img_batch, rotate_angle[0],interpolation='BILINEAR'),
    #                                      flipped_img_batch),
    #                             flipped_img_batch
    #                             )
    #tf.where里边不用lambda，不是函数，两个同size的tensor就行
    flipped_img_batch = tf.where(flip_control >= 0.5,
                                 tf.image.flip_left_right(reshaped_img_batch),
                                 reshaped_img_batch)
    rotated_img_batch = tf.where(rotate_control >=0.5,
                                 tf.contrib.image.rotate(flipped_img_batch,rotate_angle[0],interpolation = 'BILINEAR'),
                                 flipped_img_batch)#interpolation:  "NEAREST", "BILINEAR".
    #tf.cond不能用tensor？必须callable
    final_img_batch = tf.cond(is_data_augment,
                                lambda:rotated_img_batch,
                                lambda:reshaped_img_batch)

    # final_img_batch = rotated_img_batch

    test_img_batch, test_label_batch = fer_generator.get_tfrecord(MINI_BATCH, config.tfRecord_test)
    with tf.Session() as sess:
        log_dir = 'tensorboard_dir'
        test_log_dir = 'test_tensorboard_dir'
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()#
        # 继续训练
        ckpt=tf.train.get_checkpoint_state(config.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:#这个逻辑不断点续训才能有！！！
            if os.path.exists(log_dir):  # 删掉以前的summary，以免图像重合，
                shutil.rmtree(log_dir)
            os.makedirs(log_dir)
            if os.path.exists(test_log_dir):  # 删掉以前的summary，以免图像重合，
                shutil.rmtree(test_log_dir)
            os.makedirs(test_log_dir)

        writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir=test_log_dir, graph=sess.graph)
        merged = tf.summary.merge_all()
        test_merged = tf.summary.merge(inputs=[summary_cem, summary_loss, summary_accuracy])  # 不需要其他数据，只需要准确率，用上summary那个返回值

        # 创建一个线程协调器
        coord=tf.train.Coordinator()
        # 启动入队线程
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        #img_batch,label_batch=fer_generator.get_tfrecord(BATCH_SIZE,config.tfRecord_train)##这个位置是血的教训
        for i in range(STEPS):
            xs, ys = sess.run([final_img_batch,label_batch], feed_dict={is_data_augment:True})

            reshape_xs = np.reshape(xs, (BATCH_SIZE,
                                         config.img_width,
                                         config.img_height,
                                         fer_forward.NUM_CHANNELS))
            # 训练更新,accuracy_value不是很好用，考虑到dropout。
            _, loss_value, accuracy_value, step, lr = sess.run([train_op, loss, accuracy, global_step, learning_rate],
                                                           feed_dict={x: reshape_xs, y_: ys, prob:0.3, bn_training:True, is_data_augment:True})

            # writer.add_event()
            if (step + 1) % 30 == 0:
                # 训练集，避免影响训练过程拿的数据，这里不新get，使用重复数据，只是把dropout改掉。
                # 后期这里也可以换成valid数据集，如果需要
                accuracy_score, train_summary = sess.run(
                    [accuracy, merged], feed_dict={x: reshape_xs, y_: ys, prob: 1.0, bn_training:True, is_data_augment:False})# 不能直接用，必须单独run，因为dropout
                #accuracy其实应该用prob1.0，但是想收集prob信息，这样就又收集不到了（都是1.0）。只做调试用吧，无所谓了。
                writer.add_summary(train_summary, step)

                print("%s : After %d training step(s),lr is %g loss,accuracy on training batch is %g , %g." % (
                time.strftime('%Y-%m-%d %H:%M:%S'), step, lr,loss_value,accuracy_score))

                #测试集#test的准确率是分批运算的需要合成，怎么办？只能先用batch的
                xs, ys = sess.run([test_img_batch, test_label_batch])
                # reshape测试输入数据xs
                reshape_xs = np.reshape(xs, (MINI_BATCH,
                                             config.img_width,
                                             config.img_height,
                                             fer_forward.NUM_CHANNELS))

                #这个时候的测试准确率其实也不准，因为测试集也走了数据增强的预处理
                accuracy_score, test_summary = sess.run([accuracy, test_merged], feed_dict={x: reshape_xs, y_: ys, prob: 1.0, bn_training:False, is_data_augment:False})
                test_writer.add_summary(test_summary, step)
                print("After %s training step(s), test accuracy = %g" % (step, accuracy_score))

                # 输出global_step和准确率
                saver.save(sess, os.path.join(config.MODEL_SAVE_PATH, config.MODEL_NAME), global_step=global_step)

        coord.request_stop()
        coord.join(threads)

def main():
    backward()

if __name__ == '__main__':
    main()


