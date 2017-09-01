import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def avg_pool(x, size):
    return tf.nn.avg_pool(x, ksize=[1, size, 1, 1], strides=[1, size, 1, 1], padding='SAME')


def max_pool(x, size):
    return tf.nn.max_pool(x, ksize=[1, size, 1, 1], strides=[1, size, 1, 1], padding='SAME')


def get_feature(shock_value, left, right):
    before_left = max(int(left - (right - left) / 39), 0)

    right += (40 - (right - before_left) % 40) % 40

    tmp = shock_value[before_left:right].reshape(40, -1)
    tmp = np.mean(tmp, axis=1)
    tmp_max = np.max(tmp) + 1.0

    return tmp / tmp_max


def get_train_data():
    sample_range = np.load('./data/range_sample.npy')

    train_x, train_y = [], []
    for range_list in sample_range:
        unit = range_list[0]
        shock_value = np.load('./data/shock/' + unit)

        for i in np.arange(1, len(range_list)):
            left, right, flag = range_list[i]

            train_x.append(get_feature(shock_value, left, right))
            if flag == 0:
                train_y.append((1, 0))
            else:
                train_y.append((0, 1))

    return np.reshape(train_x, (-1, 40)), np.reshape(train_y, (-1, 2))


class CnnModel(object):
    def __init__(self, isTrain=False):
        self.x = tf.placeholder(tf.float32, [None, 40])  # 40
        self.x_image = tf.reshape(self.x, [-1, 40, 1, 1])
        self.y = tf.placeholder(tf.float32, [None, 2])

        # 卷积层
        self.w_conv1 = weight_variable([5, 1, 1, 20])
        self.b_conv1 = bias_variable([20])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.w_conv1) + self.b_conv1)  # 40 x 1 x 20
        self.h_pool1 = avg_pool(self.h_conv1, 2)  # 20 x 1 x 20

        # 卷积层
        self.w_conv2 = weight_variable([20, 1, 20, 40])
        self.b_conv2 = bias_variable([40])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.w_conv2) + self.b_conv2)  # 20 x 1 x 40
        self.h_pool2 = max_pool(self.h_conv2, 5)  # 4 x 1 x 40

        # 全连接层
        self.w_fc1 = weight_variable([4 * 1 * 40, 10])
        self.b_fc1 = bias_variable([10])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 4 * 1 * 40])
        self.h_fc1 = tf.nn.elu(tf.matmul(self.h_pool2_flat, self.w_fc1) + self.b_fc1)

        # 防止过拟合
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # 全连接层，输出分类结果
        self.w_fc2 = weight_variable([10, 2])
        self.b_fc2 = bias_variable([2])
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.w_fc2) + self.b_fc2)

        # 交叉熵
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.y_conv), reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # 准确度计算
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.predict_y = tf.argmax(self.y_conv, 1)

        self.sess = tf.InteractiveSession()

        saver = tf.train.Saver()
        if isTrain:
            train_x, train_y = get_train_data()

            self.sess.run(tf.initialize_all_variables())
            for i in range(1000):
                train_accuracy = self.accuracy.eval(feed_dict={self.x: train_x, self.y: train_y, self.keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))

                self.train_step.run(feed_dict={self.x: train_x, self.y: train_y, self.keep_prob: 0.7})
            saver.save(self.sess, './data/cnn/model.ckpt')
        else:
            saver.restore(self.sess, './data/cnn/model.ckpt')

    def predict(self, shock_value, left, right):
        x_value = np.array(get_feature(shock_value, left, right)).reshape(-1, 40)
        return self.predict_y.eval(feed_dict={self.x: x_value, self.y: np.reshape([0, 0], (-1, 2)), self.keep_prob: 1.0})


if __name__ == '__main__':
    cnnModel = CnnModel(isTrain=True)
