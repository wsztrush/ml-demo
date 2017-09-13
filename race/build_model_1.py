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
    right -= (right - left) % 8
    if right - left < 8:
        return None

    tmp = np.mean(shock_value[left:right].reshape(8, -1), axis=1) + 1

    return tmp / np.max(tmp)


def build_extra_before_ratio(shock_value, left, right):
    a = max(left - 10, 0)
    b = int(max((right - left) / 8, 10))
    a = np.mean(shock_value[a:left])
    b = np.mean(shock_value[left:left + b])
    return [a / b, a, b]


def build_extra_max_index(shock_value, left, right):
    a = shock_value[left:right]
    b = np.where(a == np.max(a))[0][0]
    return [b, b / (right - left)]


def build_extra_jump_index(jump_index, shock_value, left, right):
    a = shock_value[left:right]
    b = np.where(a == np.max(a))[0][0]

    before_jump_index = np.where((jump_index > left) & (jump_index < left + b))[0]
    all_jump_index = np.where((jump_index > left) & (jump_index < right))[0]

    return [len(before_jump_index), len(all_jump_index), before_jump_index, all_jump_index]


def build_extra_feature(jump_index, shock_value, left, right):
    before_ratio = build_extra_before_ratio(shock_value, left, right)
    max_index = build_extra_max_index(shock_value, left, right)
    jump_index = build_extra_jump_index(jump_index, shock_value, left, right)

    return [
        before_ratio,
        max_index,
        jump_index
    ]


def train():
    x_list = []
    for unit in os.listdir('./data/all_range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_list = np.load('./data/all_range/' + unit)

        for left, right in range_list:
            feature = build_feature(shock_value, left, right)
            if feature is not None:
                x_list.append(feature)

    print('[TOTAL]', len(x_list))

    # 训练模型
    kmeans = KMeans(n_clusters=7).fit(x_list)

    # 查看模型结果
    print(np.bincount(kmeans.labels_))
    tmp = kmeans.cluster_centers_
    print(tmp)

    plt.plot(tmp.T)
    plt.show()

    # 保存模型
    joblib.dump(kmeans, './data/model_1')


def view():
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
            shock_z_value = np.load('./data/shock_z/' + unit)
            range_list = np.load('./data/all_range/' + unit)
            origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

            for left, right in range_list:
                before_left = max(int(left - (right - left) / 9), 0)

                if race_util.range_filter(shock_value, shock_z_value, left, right):
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

    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=500, repeat=False)
    plt.show()


if __name__ == '__main__':
    # train()
    view()
    # check_extra_feature()

# [TOTAL] 238791

# [62590 39068 43855 15675 29536 29015 19052]

# [ 0.89599025  0.8632672   0.65449337  0.54023381  0.46491579  0.41236896  0.37379646  0.29616327]
# [ 0.81588216  0.7211983   0.72093472  0.73719812  0.7428436   0.73989457  0.71237998  0.53334103]
# [ 0.47311842  0.99948133  0.50864484  0.31622714  0.24099367  0.20068263  0.17270049  0.13822351]
# [ 0.50961321  0.41711334  0.3628848   0.44272846  0.8168389   0.69641961  0.45932799  0.27958546]
# [ 0.40126721  0.52182062  0.99630956  0.58162245  0.3532089   0.25567919  0.20269501  0.15750465]
# [ 0.99677703  0.53852215  0.29284859  0.21418083  0.17780447  0.15631932  0.13905171  0.10988789]
# [ 0.422303    0.36952168  0.49105132  0.99000966  0.59353381  0.36116464  0.26555574  0.19924362]
