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

        if ret == 5:
            return True

    return False


def build_feature(shock_value, left, right):
    right -= (right - left) % 10

    tmp = shock_value[left:right]
    tmp = np.mean(tmp.reshape(10, -1), axis=1)
    tmp_max = np.max(tmp) + 1.0

    return tmp / tmp_max


def train():
    x_list = []
    for unit in os.listdir('./data/all_range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_list = np.load('./data/all_range/' + unit)

        for left, right in range_list:
            x_list.append(build_feature(shock_value, left, right))

    print('[TOTAL]', len(x_list))

    # 训练模型
    kmeans = KMeans(n_clusters=10).fit(x_list)

    # 查看模型结果
    # print(kmeans.labels_)
    print(np.bincount(kmeans.labels_))
    tmp = kmeans.cluster_centers_
    print(tmp)

    plt.plot(tmp.T)
    plt.show()

    # 保存模型
    joblib.dump(kmeans, './data/model_a')


def view():
    # 加载模型
    kmeans = joblib.load('./data/model_a')

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

        for unit in unit_list:
            print(unit)
            shock_value = np.load('./data/shock/' + unit)
            range_list = np.load('./data/all_range/' + unit)
            origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

            for left, right in range_list:
                before_left = max(int(left - (right - left) / 9), 0)

                feature = build_feature(shock_value, left, right)
                predict_ret = kmeans.predict([feature])[0]
                if predict_ret == 5:
                    yield shock_value[before_left:right], origin_value[before_left * race_util.shock_step:right * race_util.shock_step]

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

    # train()
    view()

# [62941 49738 35663 27154 30236 59549 30885 40429 36241 31374]
# [[ 0.40565433  0.60361079  0.75232448  0.86989461  0.87088704  0.777011
#    0.66926861  0.57900545  0.51872594  0.46208767]
#  [ 0.32671263  0.52388058  0.94291499  0.84332425  0.64027253  0.53356365
#    0.46869335  0.42308956  0.39202626  0.35683215]
#  [ 0.24687636  0.34576719  0.47469481  0.9954606   0.62091292  0.40875832
#    0.31855348  0.26862412  0.23755434  0.20902851]
#  [ 0.52210032  0.90581785  0.47163964  0.28610851  0.21155966  0.17235776
#    0.14919566  0.13272172  0.11781031  0.1034321 ]
#  [ 0.29802361  0.37606101  0.37639932  0.38096177  0.51192874  0.90759584
#    0.72839784  0.46101115  0.35162395  0.28987184]
#  [ 0.56904391  0.7651681   0.77019877  0.75456811  0.7544828   0.78248151
#    0.79644474  0.78822789  0.76977138  0.69225322]
#  [ 0.36169899  0.46416255  0.48978285  0.49256803  0.49020691  0.53800213
#    0.72834959  0.8041499   0.67684454  0.48601234]
#  [ 0.40322853  0.94367209  0.82253573  0.63492017  0.55902312  0.5145116
#    0.48271084  0.4531068   0.42897014  0.39194194]
#  [ 0.2547534   0.50902864  0.99060959  0.57255873  0.36712848  0.28327779
#    0.23645239  0.20662082  0.18629015  0.16579042]
#  [ 0.26064803  0.34254481  0.36339222  0.51367763  0.99159286  0.63474849
#    0.41763443  0.32610225  0.27564731  0.23679097]]