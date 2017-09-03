import numpy as np
import os
import random
import race_util

from obspy import read
from matplotlib import animation
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.cluster import KMeans


def predict(model, shock_value, left, right):
    feature = build_feature(shock_value, left, right)
    if feature is not None:
        ret = model.predict([feature])[0]

        if ret in [0, 1, 2, 3, 5, 6, 8]:
            return True

    return False


def build_feature(shock_value, left, right):
    before_left = max(int(left - (right - left) / 9), 0)
    right -= (right - left) % 10

    if before_left == left:
        return None
    if right - left < 10:
        return None

    a = np.mean(shock_value[before_left:left])
    b = np.mean(shock_value[left:right].reshape(10, -1), axis=1)

    return [a / (b[0] + 1.0)]


def train():
    x_list = []
    for unit in os.listdir('./data/range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_list = np.load('./data/range/' + unit)

        for left, right in range_list:
            feature = build_feature(shock_value, left, right)
            if feature is not None:
                x_list.append(feature)

    print('[TOTAL]', len(x_list))

    # 训练模型
    kmeans = KMeans(n_clusters=10).fit(x_list)

    # 查看模型结果
    print(np.bincount(kmeans.labels_))
    tmp = kmeans.cluster_centers_
    print(tmp)

    plt.plot(tmp.T)
    plt.show()

    # 保存模型
    joblib.dump(kmeans, './data/model_b')


def view():
    # 加载模型
    kmeans = joblib.load('./data/model_b')

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
                if feature is None:
                    continue
                predict_ret = kmeans.predict([feature])[0]
                if predict_ret == 8:
                    yield shock_value[before_left:right], origin_value[before_left * race_util.shock_step:right * race_util.shock_step]

    def refresh(value):
        line1.set_data(np.arange(len(value[0])), value[0])
        line2.set_data(np.arange(len(value[1])), value[1])

        ax1.set_ylim(np.min(value[0]), np.max(value[0]))
        ax1.set_xlim(0, len(value[0]))

        ax2.set_ylim(np.min(value[1]), np.max(value[1]))
        ax2.set_xlim(0, len(value[1]))

        return line1, line2

    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=100, repeat=False)
    plt.show()


if __name__ == '__main__':
    race_util.config()

    # train()
    view()

# [ 8836     2 95961     2 56215     2   689 61297 47633 74123]
# [[  1.71211812e+00] * 0
#  [  2.44025970e+02] * 1
#  [  9.33024266e-01] * 2
#  [  3.93839355e+02] * 3
#  [  4.20194770e-01]
#  [  1.53816071e+02] * 5
#  [  3.26726288e+00] * 6
#  [  1.28135878e-01]
#  [  1.18766524e+00] * 8
#  [  6.98271380e-01]
# ]