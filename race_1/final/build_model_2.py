import numpy as np
import os
import random
import race_util

from obspy import read
from matplotlib import animation
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.cluster import KMeans


def build_feature(shock_value, left, right):
    right -= (right - left) % 10

    tmp = shock_value[left:right]
    tmp = np.mean(tmp.reshape(10, -1), axis=1)
    tmp_max = np.max(tmp) + 1.0

    return tmp / tmp_max


def train():
    x_list = []
    for unit in os.listdir('./data/range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_list = np.load('./data/range/' + unit)

        for left, right in range_list:
            x_list.append(build_feature(shock_value, left, right))

    # 训练模型
    kmeans = KMeans(n_clusters=20).fit(x_list)

    # 查看模型结果
    print(kmeans.labels_)
    print(np.bincount(kmeans.labels_))
    tmp = kmeans.cluster_centers_
    print(tmp)

    plt.plot(tmp.T)
    plt.show()

    # 保存模型
    joblib.dump(kmeans, './data/model_2')


def view():
    # 加载模型
    kmeans = joblib.load('./data/model_2')

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.set_ylim(0, 10000)
    ax1.set_xlim(0, 10)

    ax2 = fig.add_subplot(212)
    ax2.set_ylim(0, 10000)
    ax2.set_xlim(0, 10)

    line1, = ax1.plot([], [])
    line2, = ax2.plot([], [])

    def next_value():
        unit_list = os.listdir('./data/range/')
        random.shuffle(unit_list)

        for unit in unit_list:
            print(unit)
            shock_value = np.load('./data/shock/' + unit)
            range_list = np.load('./data/range/' + unit)
            origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

            for left, right in range_list:
                before_left = max(int(left - (right - left) / 9), 0)

                feature = build_feature(shock_value, left, right)
                predict_ret = kmeans.predict([feature])
                if predict_ret == 19:
                    yield shock_value[before_left:right], origin_value[before_left * race_util.shock_step:right * race_util.shock_step]

    def refresh(value):
        line1.set_data(np.arange(len(value[0])), value[0])
        line2.set_data(np.arange(len(value[1])), value[1])

        ax1.set_ylim(np.min(value[0]), np.max(value[0]))
        ax1.set_xlim(0, len(value[0]))

        ax2.set_ylim(np.min(value[1]), np.max(value[1]))
        ax2.set_xlim(0, len(value[1]))

        return line1, line2

    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=200, repeat=False)
    plt.show()


if __name__ == '__main__':
    race_util.config()

    # train()
    view()
