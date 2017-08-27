import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_graph():
    x = tf.placeholder(tf.float32, [None, 20])
    y = tf.placeholder(tf.float32, [None, 1])

    # 卷积层
    w_conv1 = weight_variable([5, 1, 2, 10])
    b_conv1 = bias_variable([10])
    h_conv1 = tf.nn.elu(conv2d(x, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 全连接
    w_fc1 = weight_variable([10, 2])
    b_fc1 = bias_variable([2])
    h_fc1 = tf.nn.elu(tf.matmul(h_pool1, w_fc1) + b_fc1)

    # 防止过拟合
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



sess = tf.InteractiveSession()
