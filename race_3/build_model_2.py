import numpy as np
import os
import random
import race_util
import build_model_1

from obspy import read
from matplotlib import animation
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.cluster import KMeans

model_1 = joblib.load('./data/model_1')


def build_feature(shock_value, left, right):
    before_left = max(left - 10, 0)
    if before_left == left:
        return None

    right -= (right - left) % 8
    if right - left < 8:
        return None

    tmp = np.mean(shock_value[left:right].reshape(8, -1), axis=1) + 1
    tmp_max = np.max(tmp)
    return [np.mean(shock_value[before_left:left]) / tmp_max] + (tmp / tmp_max).tolist()


def train():
    x_dict = dict()
    for unit in os.listdir('./data/all_range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_list = np.load('./data/all_range/' + unit)

        for left, right in range_list:
            feature_1 = build_model_1.build_feature(shock_value, left, right)

            if feature_1 is None:
                continue

            c = model_1.predict([feature_1])[0]
            l = x_dict.get(c)
            if l is None:
                l = x_dict[c] = []

            feature_2 = build_feature(shock_value, left, right)
            l.append(feature_2)

    for i in np.arange(10):
        print('[CLASS] = ', i, '[LEN] = ', len(x_dict[i]))
        kmeans = KMeans(n_clusters=10).fit(x_dict[i])

        print(np.bincount(kmeans.labels_))
        tmp = kmeans.cluster_centers_
        print(tmp)

        plt.plot(tmp.T)
        plt.show()

        joblib.dump(kmeans, './data/model_2_' + str(i))


def view():
    # 加载模型
    model_2 = [joblib.load('./data/model_2_' + str(i)) for i in np.arange(10)]

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
            origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHZ')[0].data

            for left, right in range_list:
                before_left = max(int(left - (right - left) / 9), 0)

                feature_1 = build_model_1.build_feature(shock_value, left, right)
                feature_2 = build_feature(shock_value, left, right)

                if feature_1 is None or feature_2 is None:
                    continue

                predict_1 = model_1.predict([feature_1])[0]
                predict_2 = model_2[predict_1].predict([feature_2])[0]

                if predict_1 == 0 and predict_2 == 0:
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


def check():
    model_2 = [joblib.load('./data/model_2_' + str(i)) for i in np.arange(10)]

    c = [0, 0, 0]

    for unit in os.listdir('./data/all_range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_list = np.load('./data/all_range/' + unit)

        for left, right in range_list:
            feature_1 = build_model_1.build_feature(shock_value, left, right)
            feature_2 = build_feature(shock_value, left, right)

            if feature_1 is None or feature_2 is None:
                continue

            predict_1 = model_1.predict([feature_1])[0]
            predict_2 = model_2[predict_1].predict([feature_2])[0]

            c[flag[predict_1][predict_2]] += 1

    print(c)


if __name__ == '__main__':
    race_util.config()
    train()
    # view()
    # check()

flag = [
]
# [CLASS] =  0 [LEN] =  20980
# [2726 1897 2378  416 2746 2178 2241 1502 2249 2647]
# [[ 0.11280507]
#  [ 0.40659915]
#  [ 0.25838208]
#  [ 0.52821812]
#  [ 0.16106287]
#  [ 0.06150536]
#  [ 0.35689485]
#  [ 0.46042104]
#  [ 0.30710173]
#  [ 0.2091147 ]]
# [CLASS] =  1 [LEN] =  24827
# [1293 2827 2938 2328 2994 2407 1329 3187 2903 2621]
# [[ 0.48153604]
#  [ 0.20758508]
#  [ 0.33962343]
#  [ 0.11489817]
#  [ 0.25300821]
#  [ 0.42861231]
#  [ 0.0610313 ]
#  [ 0.29753102]
#  [ 0.38185214]
#  [ 0.16182154]]
# [CLASS] =  2 [LEN] =  11303
# [1554 1652  535 1808 1731  526  894 1157  270 1176]
# [[ 0.3103423 ]
#  [ 0.44337252]
#  [ 0.14614343]
#  [ 0.35432839]
#  [ 0.39761791]
#  [ 0.55743745]
#  [ 0.209813  ]
#  [ 0.49135898]
#  [ 0.0621567 ]
#  [ 0.26225785]]
# [CLASS] =  3 [LEN] =  21655
# [3863 1070 2138  378 1265 3392  708 4724 1607 2510]
# [[ 0.02060604]
#  [ 0.14937433]
#  [ 0.07167863]
#  [ 0.23165245]
#  [ 0.12011996]
#  [ 0.03600737]
#  [ 0.18437043]
#  [ 0.0067227 ]
#  [ 0.09410914]
#  [ 0.05237455]]
# [CLASS] =  4 [LEN] =  28254
# [4222 2564  959 4684 1980 4114 4996 1297  291 3147]
# [[ 0.355501  ]
#  [ 0.51492755]
#  [ 0.18895758]
#  [ 0.43242265]
#  [ 0.25900984]
#  [ 0.47140029]
#  [ 0.39457416]
#  [ 0.57005039]
#  [ 0.08504857]
#  [ 0.3120639 ]]
# [CLASS] =  5 [LEN] =  16541
# [1534 1853 2133 2064  473 2068  862 1844 1513 2197]
# [[ 0.13889324]
#  [ 0.41661563]
#  [ 0.28088814]
#  [ 0.23631617]
#  [ 0.54136834]
#  [ 0.3692923 ]
#  [ 0.07748515]
#  [ 0.18974045]
#  [ 0.46859324]
#  [ 0.32454432]]
# [CLASS] =  6 [LEN] =  33751
# [4854 5196 1492 3171  923 4110 2345 2668 4016 4976]
# [[ 0.24377444]
#  [ 0.27760574]
#  [ 0.07091378]
#  [ 0.16855557]
#  [ 0.44489864]
#  [ 0.34437984]
#  [ 0.12432563]
#  [ 0.38529347]
#  [ 0.20700466]
#  [ 0.31027017]]
# [CLASS] =  7 [LEN] =  27223
# [4405 2620 1588 3498  560 3061 4031 4198 1177 2085]
# [[ 0.06341471]
#  [ 0.23467258]
#  [ 0.34010605]
#  [ 0.14361491]
#  [ 0.47447299]
#  [ 0.18802637]
#  [ 0.10198138]
#  [ 0.02683483]
#  [ 0.39906942]
#  [ 0.28673422]]
# [CLASS] =  8 [LEN] =  26070
# [2612 2925 3968 1404 4115 2182 3739 3432  873  820]
# [[ 0.4646281 ]
#  [ 0.27498337]
#  [ 0.38642981]
#  [ 0.17770069]
#  [ 0.35018469]
#  [ 0.23130805]
#  [ 0.31369179]
#  [ 0.42445654]
#  [ 0.51638006]
#  [ 0.10776988]]
# [CLASS] =  9 [LEN] =  13762
# [1287 1933 1931  827  548 1882 1955 1646 1369  384]
# [[ 0.21351075]
#  [ 0.43854136]
#  [ 0.3530752 ]
#  [ 0.15095671]
#  [ 0.54859509]
#  [ 0.30959901]
#  [ 0.3956528 ]
#  [ 0.26383981]
#  [ 0.48443444]
#  [ 0.06765573]]
