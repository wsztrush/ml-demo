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
    if before_left == left:
        return None

    right -= (right - left) % 9
    if right - left < 9:
        return None

    a = np.mean(shock_value[before_left:left])
    b = np.mean(shock_value[left:right].reshape(9, -1), axis=1) + 1
    b_max = np.max(b)

    ret = [a / b_max] + (b / b_max).tolist()
    return ret


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
    joblib.dump(kmeans, './data/model_1')


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
        unit_list = os.listdir('./data/range/')
        random.shuffle(unit_list)
        total = 0

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
                if predict_ret == 9:
                    total += 1
                    print(total)
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

# [TOTAL] 306776
# [31090 41589 27832 22910 21091 41691 30372  8876 54171 27154]
# [[ 0.10782827  0.26870568  0.48633096  0.99674125  0.56026741  0.35410717
#    0.26978236  0.2236691   0.19452441  0.17199022]
#  [ 0.23989631  0.33905871  0.5399548   0.95370481  0.83121239  0.6216383
#    0.50910616  0.43994469  0.3929111   0.35688397]
#  [ 0.08454858  0.37583298  0.99532371  0.57268848  0.34465668  0.25459047
#    0.20735624  0.17874599  0.15697318  0.13733192]
#  [ 0.29262985  0.40359155  0.49120128  0.51406138  0.52399524  0.54021849
#    0.63178542  0.82113055  0.78791772  0.59747619]
#  [ 0.17674907  0.32836083  0.39245129  0.38252993  0.38974281  0.5586906
#    0.95418215  0.66468547  0.42436912  0.32997137]
#  [ 0.28694075  0.43699476  0.9483072   0.83543446  0.6448683   0.56348882
#    0.51606553  0.48112526  0.45121114  0.42082436]
#  [ 0.13180892  0.26927497  0.35468816  0.47013643  0.9969395   0.6166378
#    0.39922219  0.30538989  0.25466794  0.22091175]
#  [ 0.0517984   0.95707392  0.63843371  0.29838446  0.20151757  0.15902915
#    0.13588886  0.12093151  0.1048527   0.08955548]
#  [ 0.33620018  0.43782748  0.6478878   0.79498291  0.89139695  0.85725584
#    0.75237827  0.64320522  0.5625873   0.50695242]
#  [ 0.15192633  0.2948505   0.36281746  0.36801189  0.52995505  0.99630527
#    0.62411951  0.41514069  0.32184284  0.26989928]]