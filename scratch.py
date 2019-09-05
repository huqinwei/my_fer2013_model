import tensorflow as tf

#0不能作为筛选，但是乘以0,可以避免不想正则的被正则。
w1 = 0
w1 = None
if w1 is not None:
    print('hello')

#也没关系，因为w1==0，乘以w1，就等于清零了。
# weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name=name + '_weight_loss')

a = tf.Variable(tf.constant([12.,3]))
b = tf.multiply(tf.nn.l2_loss(a),0)
c = tf.multiply(tf.nn.l2_loss(a),0.1)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

