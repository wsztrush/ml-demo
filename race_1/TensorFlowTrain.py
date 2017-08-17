import os
import numpy as np
from obspy import read
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
import tensorflow as tf
import math

DIR_PATH = "/Users/tianchi.gzt/Downloads/example30/"
INTERVAL = 5


def process(filepath):
    file_conent = read(filepath)

    b = file_conent[0].meta.sac['b']
    s = (file_conent[0].meta.sac['t0'] - b) * 100
    p = (file_conent[0].meta.sac['a'] - b) * 100

    # print(s, p)

    data = file_conent[0].data
    data = data[:len(data) - len(data) % INTERVAL]
    data = data.reshape(-1, INTERVAL)

    y = [np.std(data, axis=1)][0]
    x = np.arange(0, len(y)).reshape(-1, 1)

    # Sklearn训练
    #
    # clf = linear_model.LinearRegression()
    # clf = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
    # clf = SVR(kernel='rbf', degree=2)
    # clf = SVR(C=1e-5, epsilon=0.1)
    # clf.fit(x, y)
    # print(clf)
    #
    # for i in np.arange(len(y)):
    #     predict_y[i] = clf.predict(i)

    # print(predict_y)

    # TensorFlow训练
    # y = b +
    #     s_1 * e^(-s_2 * (x-s)) * (x-s) ^ s_3 * (sign(x-s) + 1) +
    #     p_1 * e^(-p_2 * (x-p)) * (x-p) ^ p_3 * (sign(x-p) + 1)
    # b = tf.Variable([20.], tf.float32)
    # s = tf.Variable([98.], tf.float32)
    # s_1 = tf.Variable([10.], tf.float32)
    # s_2 = tf.Variable([-.01], tf.float32)
    # s_3 = tf.Variable([.01], tf.float32)
    # p = tf.Variable([float(len(y))], tf.float32)
    # p_1 = tf.Variable([100.], tf.float32)
    # p_2 = tf.Variable([-.01], tf.float32)
    # p_3 = tf.Variable([.01], tf.float32)
    # #
    # tx = tf.placeholder(tf.float32)
    # ty = tf.placeholder(tf.float32)
    # #
    # f = b + \
    #     s_1 * math.e ** (s_2 * (tx - s)) * (tf.sigmoid(tx - s)) ** s_3 + \
    #     p_1 * math.e ** (p_2 * (tx - p)) * (tf.sigmoid(tx - p)) ** p_3

    # loss = tf.reduce_sum(tf.log(f - ty))
    # optimizer = tf.train.GradientDescentOptimizer(1e-6)
    # train = optimizer.minimize(loss)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     for i in range(2):
    #         sess.run(train, {tx: x, ty: y})
    #
    #         print(sess.run([b, s, s_1, s_2, s_3, p, p_1, p_2, p_3]))


    plt.plot(x, y)
    if p > 0:
        plt.plot(p / INTERVAL, 0, 'ro')
    if s > 0:
        plt.plot(s / INTERVAL, 0, 'yo')
    plt.show()


if __name__ == '__main__':
    for filename in os.listdir(DIR_PATH):
        process(DIR_PATH + filename)

        # break
