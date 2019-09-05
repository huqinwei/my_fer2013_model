#coding:utf-8

#lenet-5改，其实更像照着alexnet改的，如果是alexnet，norm全在pooling之后
#bias_add是可以重名的，但是如果不改名，图是断的。另一方面，restore模型也不是很方便，出问题不好调。
#第三个版本，dropout做完后，不急调整dropout率，开始改LRN为BN
#把LRN改成BN，BN放激活之前。
#继续降低Dropout率
#提升batch

#程序结构上，dropout看起来不影响结构吧？relu3还是relu3，但是我的bn层true和false两个效果，选择不训练，效果不好，可能训练不充分
import tensorflow as tf

NUM_CHANNELS=1
CONV1_KSIZE=5
CONV1_OUTPUT_CHANNEL=64
CONV2_KSIZE=5
CONV2_OUTPUT_CHANNEL=64
CONV3_KSIZE=5
CONV3_OUTPUT_CHANNEL=128




FC_SIZE=512

#输出，7分类
OUTPUT_NODE=7


#形状，数据stddev，正则化系数
def variable_with_weight_loss(shape, std, w1,name):
    var = tf.Variable(tf.truncated_normal(shape, stddev = std), dtype=tf.float32, name = name)
    if w1 is not None:
        #这里不给每一个变量单独命名的话，graph看就是一个重名数组，有很多断点
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name=name+'_weight_loss')
        tf.add_to_collection('regularization_losses', weight_loss)
    return var

# 卷积层
def conv2d(x, w,name):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME',name=name)
#池化层:带重叠的
def max_pooling_2x2(x,name):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME',name=name)



def forward(x, keep_prob,bn_training,bn_enable = False):
    return_dict = dict()
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    conv1_w = variable_with_weight_loss(
        shape=[CONV1_KSIZE,CONV1_KSIZE,NUM_CHANNELS,CONV1_OUTPUT_CHANNEL],std=0.05,w1=0,name='conv1_w')
    conv1_b = tf.Variable(tf.constant(0.1, dtype = tf.float32, shape=[CONV1_OUTPUT_CHANNEL]),name='conv1_b')
    conv1 = conv2d(x,conv1_w,'conv1')
    # conv1_plus_b = tf.nn.bias_add(conv1,conv1_b,name='conv1_plus_b')
    conv1_plus_b = conv1
    if bn_enable:
        bn1 = tf.layers.batch_normalization(conv1_plus_b,name='bn1',training=bn_training)#bn_training可以是python和tensor的boolean
        print(tf.global_variables())
        tf.summary.histogram('outs/bn1', bn1)
        gamma = tf.get_default_graph().get_tensor_by_name(name='bn1/gamma:0')
        beta = tf.get_default_graph().get_tensor_by_name(name='bn1/beta:0')
        moving_mean = tf.get_default_graph().get_tensor_by_name(name='bn1/moving_mean:0')
        moving_variance = tf.get_default_graph().get_tensor_by_name(name='bn1/moving_variance:0')

        tf.summary.histogram('bn1/gamma', gamma)
        tf.summary.histogram('bn1/beta', beta)
        tf.summary.histogram('bn1/moving_mean', moving_mean)
        tf.summary.histogram('bn1/moving_variance', moving_variance)
        relu1 = tf.nn.relu(bn1, name='relu1')
    else:
        relu1 = tf.nn.relu(conv1_plus_b, name='relu1')
    return_dict['relu1_output'] = relu1


    pool1 = max_pooling_2x2(relu1,name='pool1')
    return_dict['pool1_output'] = pool1

    tf.summary.histogram('weights/conv1_w', conv1_w)
    tf.summary.histogram('weights/conv1_b', conv1_b)

    tf.summary.histogram('outs/conv1', conv1)
    tf.summary.histogram('outs/relu1', relu1)
    tf.summary.histogram('outs/pool1', pool1)
    
    conv2_w = variable_with_weight_loss(
        shape=[CONV2_KSIZE,CONV2_KSIZE,CONV1_OUTPUT_CHANNEL, CONV2_OUTPUT_CHANNEL],std=0.05,w1=0,name='conv2_w')
    conv2_b = tf.Variable(tf.constant(0.1, dtype = tf.float32, shape=[CONV2_OUTPUT_CHANNEL]),name='conv2_b')
    conv2 = conv2d(pool1, conv2_w,'conv2')
    # conv2_plus_b = tf.nn.bias_add(conv2,conv2_b,name='conv2_plus_b')
    conv2_plus_b = conv2
    if bn_enable:
        bn2 = tf.layers.batch_normalization(conv2_plus_b,name='bn2',training=bn_training)
        tf.summary.histogram('outs/bn2', bn2)
        relu2 = tf.nn.relu(bn2, name='relu2')
        gamma = tf.get_default_graph().get_tensor_by_name(name='bn2/gamma:0')
        beta = tf.get_default_graph().get_tensor_by_name(name='bn2/beta:0')
        moving_mean = tf.get_default_graph().get_tensor_by_name(name='bn2/moving_mean:0')
        moving_variance = tf.get_default_graph().get_tensor_by_name(name='bn2/moving_variance:0')

        tf.summary.histogram('bn2/gamma', gamma)
        tf.summary.histogram('bn2/beta', beta)
        tf.summary.histogram('bn2/moving_mean', moving_mean)
        tf.summary.histogram('bn2/moving_variance', moving_variance)
    else:
        relu2 = tf.nn.relu(conv2_plus_b, name='relu2')
    return_dict['relu2_output'] = relu2

    pool2 = max_pooling_2x2(relu2,name='pool2')
    return_dict['pool2_output'] = pool2

    tf.summary.histogram('weights/conv2_w', conv2_w)
    tf.summary.histogram('weights/conv2_b', conv2_b)

    tf.summary.histogram('outs/conv2', conv2)
    tf.summary.histogram('outs/relu2', relu2)
    tf.summary.histogram('outs/pool2', pool2)

    conv3_w = variable_with_weight_loss(
        shape=[CONV3_KSIZE, CONV3_KSIZE,CONV2_OUTPUT_CHANNEL,CONV3_OUTPUT_CHANNEL],std=0.05,w1=0,name='conv3_w')
    conv3_b = tf.Variable(tf.constant(0.01, dtype = tf.float32, shape=[CONV3_OUTPUT_CHANNEL]),name='conv3_b')
    conv3 = conv2d(pool2, conv3_w,'conv1')
    # conv3_plus_b = tf.nn.bias_add(conv3,conv3_b,name='conv3_plus_b')
    conv3_plus_b = conv3
    if bn_enable:
        bn3 = tf.layers.batch_normalization(conv3_plus_b,name='bn3',training=bn_training)
        tf.summary.histogram('outs/bn3', bn3)
        relu3 = tf.nn.relu(bn3, name='relu3')

        gamma = tf.get_default_graph().get_tensor_by_name(name='bn3/gamma:0')
        beta = tf.get_default_graph().get_tensor_by_name(name='bn3/beta:0')
        moving_mean = tf.get_default_graph().get_tensor_by_name(name='bn3/moving_mean:0')
        moving_variance = tf.get_default_graph().get_tensor_by_name(name='bn3/moving_variance:0')

        tf.summary.histogram('bn3/gamma', gamma)
        tf.summary.histogram('bn3/beta', beta)
        tf.summary.histogram('bn3/moving_mean', moving_mean)
        tf.summary.histogram('bn3/moving_variance', moving_variance)
    else:
        relu3 = tf.nn.relu(conv3_plus_b, name='relu3')
    return_dict['relu3_output'] = relu3
    pool3 = max_pooling_2x2(relu3,name='pool3')
    return_dict['pool3_output'] = pool3

    tf.summary.histogram('weights/conv3_w', conv3_w)
    tf.summary.histogram('weights/conv3_b', conv3_b)

    tf.summary.histogram('outs/conv3', conv3)
    tf.summary.histogram('outs/relu3', relu3)
    tf.summary.histogram('outs/pool3', pool3)

    pool3 = tf.nn.dropout(pool3, keep_prob)
    #FC层
    pool3_shape = pool3.get_shape().as_list()
    nodes = pool3_shape[1] * pool3_shape[2] * pool3_shape[3]
    reshaped = tf.reshape(pool3, [-1, nodes])
    tf.summary.histogram('outs/pool3_reshaped', reshaped)
    
    w1 = variable_with_weight_loss(shape=[nodes,FC_SIZE],std=0.04,w1=0.008,name='fc1_w')#0.004
    b1 = tf.Variable(tf.constant(0.1, shape=[FC_SIZE], dtype=tf.float32),name='fc1_b')

    # fc1_plus_b = tf.nn.bias_add(tf.matmul(reshaped, w1), b1, name='bias_add_before_y1')
    fc1_plus_b = tf.matmul(reshaped, w1)

    if bn_enable:
        bn_fc1 = tf.layers.batch_normalization(fc1_plus_b,name='bn_fc1',training=bn_training)
        tf.summary.histogram('outs/bn_fc1', bn_fc1)
        y1 = tf.nn.relu(bn_fc1, name='y1')
        gamma = tf.get_default_graph().get_tensor_by_name(name='bn_fc1/gamma:0')
        beta = tf.get_default_graph().get_tensor_by_name(name='bn_fc1/beta:0')
        moving_mean = tf.get_default_graph().get_tensor_by_name(name='bn_fc1/moving_mean:0')
        moving_variance = tf.get_default_graph().get_tensor_by_name(name='bn_fc1/moving_variance:0')

        tf.summary.histogram('bn_fc1/gamma', gamma)
        tf.summary.histogram('bn_fc1/beta', beta)
        tf.summary.histogram('bn_fc1/moving_mean', moving_mean)
        tf.summary.histogram('bn_fc1/moving_variance', moving_variance)
    else:
        y1 = tf.nn.relu(fc1_plus_b, name='y1')
    return_dict['y1_output'] = y1

    y1 = tf.nn.dropout(y1,keep_prob)
    tf.summary.histogram('outs/y1', y1)
    
    w2 = variable_with_weight_loss(shape=[FC_SIZE, FC_SIZE], std=0.01, w1=0.008,name='fc2_w')#0.004
    b2 = tf.Variable(tf.constant(0.1,shape=[FC_SIZE], dtype=tf.float32),name='fc2_b')


    fc2_plus_b = tf.matmul(y1, w2)#用了BN，暂时不加biases

    if bn_enable:
        bn_fc2 = tf.layers.batch_normalization(fc2_plus_b,name='bn_fc2',training=bn_training)
        tf.summary.histogram('outs/bn_fc2', bn_fc2)
        y2 = tf.nn.relu(bn_fc2, name='y2')
        gamma = tf.get_default_graph().get_tensor_by_name(name='bn_fc2/gamma:0')
        beta = tf.get_default_graph().get_tensor_by_name(name='bn_fc2/beta:0')
        moving_mean = tf.get_default_graph().get_tensor_by_name(name='bn_fc2/moving_mean:0')
        moving_variance = tf.get_default_graph().get_tensor_by_name(name='bn_fc2/moving_variance:0')

        tf.summary.histogram('bn_fc2/gamma', gamma)
        tf.summary.histogram('bn_fc2/beta', beta)
        tf.summary.histogram('bn_fc2/moving_mean', moving_mean)
        tf.summary.histogram('bn_fc2/moving_variance', moving_variance)
    else:
        y2 = tf.nn.relu(fc2_plus_b, name='y2')
    return_dict['y2_output'] = y2


    y2 = tf.nn.dropout(y2,keep_prob)
    tf.summary.histogram('outs/y2', y2)

    
    w3 = variable_with_weight_loss(shape=[FC_SIZE,OUTPUT_NODE], std=1.0 / FC_SIZE, w1 = 0,name='fc3_w')#这层不正则
#     b3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE], dtype=tf.float32))#差别大吗？
    b3 = tf.Variable(tf.zeros(shape=[OUTPUT_NODE], dtype=tf.float32),name='fc3_b')
    y3 = tf.add(tf.matmul(y2,w3) ,b3, name='prediction')
    tf.summary.histogram('outs/y3', y3)

    tf.summary.histogram('weights/fc_w1', w1)
    tf.summary.histogram('weights/fc_b1', b1)
    tf.summary.histogram('weights/fc_w2', w2)
    tf.summary.histogram('weights/fc_b2', b2)
    tf.summary.histogram('weights/fc_w3', w3)
    tf.summary.histogram('weights/fc_b3', b3)

    return y3, return_dict


