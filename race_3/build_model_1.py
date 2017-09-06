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
    before_left = max(left - 10, 0)
    if before_left == left:
        return None

    right -= (right - left) % 8
    if right - left < 8:
        return None

    a = np.mean(shock_value[before_left:left])
    b = np.mean(shock_value[left:right].reshape(8, -1), axis=1) + 1
    b_max = np.max(b)

    return [
        np.mean(b[-3:]) / np.mean(b[:-3])
    ]


def train():
    x_list = []
    t = []
    for unit in os.listdir('./data/all_range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_list = np.load('./data/all_range/' + unit)

        for left, right in range_list:
            feature = build_feature(shock_value, left, right)
            if feature is not None:
                x_list.append(feature)
                t.append(feature[0])

    print('[TOTAL]', len(x_list))

    print(np.histogram(t, bins=10, range=(0, 2)))

    # 训练模型
    # kmeans = KMeans(n_clusters=10).fit(x_list)

    # 查看模型结果
    # print(np.bincount(kmeans.labels_))
    # tmp = kmeans.cluster_centers_
    # print(tmp)
    #
    # plt.plot(tmp.T)
    # plt.show()
    #
    # # 保存模型
    # joblib.dump(kmeans, './data/model_1')


def view():
    # 加载模型
    kmeans = joblib.load('./data/model_1')

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
        unit_list = os.listdir('./data/all_range/')
        random.shuffle(unit_list)
        total = 0

        for unit in unit_list:
            print(unit)
            shock_value = np.load('./data/shock/' + unit)
            range_list = np.load('./data/all_range/' + unit)
            origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

            for left, right in range_list:
                before_left = max(int(left - (right - left) / 9), 0)

                feature = build_feature(shock_value, left, right)
                if feature is None:
                    continue
                predict_ret = kmeans.predict([feature])[0]
                if predict_ret == 2:
                    total += 1
                    print(total)
                    yield shock_value[before_left:right], origin_value[before_left * race_util.step:right * race_util.step]

    def refresh(value):
        line1.set_data(np.arange(len(value[0])), value[0])
        line2.set_data(np.arange(len(value[1])), value[1])

        ax1.set_ylim(np.min(value[0]), np.max(value[0]))
        ax1.set_xlim(0, len(value[0]))

        ax2.set_ylim(np.min(value[1]), np.max(value[1]))
        ax2.set_xlim(0, len(value[1]))

        return line1, line2

    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=300, repeat=False)
    plt.show()


if __name__ == '__main__':
    race_util.config()

    train()
    # view()

# [39591 42947 32564 40482 38911 49610 39670 38769 23928 37325]
# [[ 0.15781823]
#  [ 0.04225241]
#  [ 0.24432132]
#  [ 0.07206038]
#  [ 0.1855554 ]
#  [ 0.0131096 ]
#  [ 0.10141478]
#  [ 0.12994839]
#  [ 0.27817741]
#  [ 0.214152  ]]
