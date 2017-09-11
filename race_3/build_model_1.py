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
            range_list = np.load('./data/all_range/' + unit)
            origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

            for left, right in range_list:
                before_left = max(int(left - (right - left) / 9), 0)

                feature = build_feature(shock_value, left, right)
                if feature is None:
                    continue
                predict_ret = kmeans.predict([feature])[0]

                if predict_ret == 4 and race_util.filter_4(shock_value, left, right):
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

    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=200, repeat=False)
    plt.show()


def check_extra_feature():
    model_1 = joblib.load('./data/model_1')

    x_list = dict()
    for unit in os.listdir('./data/all_range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_list = np.load('./data/all_range/' + unit)
        jump_index = np.load('./data/jump/' + unit)

        for left, right in range_list:
            feature = build_feature(shock_value, left, right)
            extra_feature = build_extra_feature(jump_index, shock_value, left, right)

            if feature is None:
                continue

            c = model_1.predict([feature])[0]
            l = x_list.get(c)
            if l is None:
                l = x_list[c] = []

            l.append(right - left)

    for i in np.arange(7):
        print(i, np.histogram(x_list[i], bins=5, range=(50, 100)))
        plt.hist(x_list[i], bins=5, range=(50, 100))
        plt.show()


if __name__ == '__main__':
    # train()
    view()
    # check_extra_feature()

# [TOTAL] 238791
# [62605 29529 43856 15696 29034 19073 38998]
# [[ 0.89584541  0.8632947   0.65477086  0.54048548  0.46510977  0.41252181
#    0.37396528  0.29628282]
#  [ 0.40115803  0.52178753  0.99634265  0.58134721  0.35303778  0.25555288
#    0.20256838  0.15740007]
#  [ 0.47318028  0.9994939   0.50854158  0.31621624  0.24099865  0.20070666
#    0.17272105  0.13823306]
#  [ 0.51037815  0.41744849  0.36313959  0.44250483  0.8161885   0.69664994
#    0.46010648  0.279985  ]
#  [ 0.99679585  0.53858942  0.29303432  0.21432466  0.17796501  0.15647025
#    0.13920035  0.11001444]
#  [ 0.42249139  0.3696907   0.49128329  0.98991289  0.59376699  0.36129939
#    0.26574394  0.19934205]
#  [ 0.81594729  0.72135365  0.72121604  0.73750304  0.74312264  0.74017835
#    0.71262448  0.53367587]]
