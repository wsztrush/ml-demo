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

    right -= (right - left) % 5
    if right - left < 5:
        return None

    a = np.mean(shock_value[before_left:left])
    b = np.mean(shock_value[left:right].reshape(5, -1), axis=1) + 1.0
    b_max = np.max(b)

    return [
        a / b[0],
        a / b_max,
    ]


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

                if predict_1 == 3 and predict_2 == 3:
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


flag = [
    [2, 0, 0, 0, 1, 2, 0, 0, 0, 1],
    [2, 1, 0, 0, 0, 1, 2, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 1, 0, 1, 0, 2, 1, 1],
    [2, 2, 1, 1, 1, 2, 2, 1, 0, 2],
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 2, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 0, 0, 2],
]
if __name__ == '__main__':
    # race_util.config()
    # train()
    view()
    # check()



# [CLASS] =  0 [LEN] =  47957
# [3018 6956 6685 4459 5545 1675 7175 6147 1953 4344]
# [[ 0.14544587  0.14518563] 2
#  [ 0.34556707  0.3451077 ] 0
#  [ 0.27501813  0.27457319] 0
#  [ 0.42093465  0.42037113] 0
#  [ 0.23687783  0.23645014] 0
#  [ 0.0830691   0.08291826] 2
#  [ 0.31065367  0.31022105] 0
#  [ 0.3815275   0.38103871] 0
#  [ 0.47249989  0.47175608] 0
#  [ 0.19443949  0.19405457] 0
# ]
# [CLASS] =  1 [LEN] =  21457
# [2814 2467  639 1192   62 2707 4729    1 4896 1950]
# [[ 0.1325113   0.06971042] 2
#  [ 0.47506637  0.15860503] 1
#  [ 1.13836842  0.19246276] 0
#  [ 0.86444166  0.2244256 ] 0
#  [ 1.79623168  0.17711046] 0
#  [ 0.44153096  0.29560274] 0 ? 1
#  [ 0.24409522  0.11783854] 2
#  [ 8.72614574  0.09933525] 0
#  [ 0.34670227  0.15864829] 1
#  [ 0.65368879  0.25213127] 2 ? 0
# ]
# [CLASS] =  2 [LEN] =  47554
# [2931 7641 7058 3558  830 8765 3219 3948 4719 4885]
# [[ 0.26426486  0.208457  ] 0
#  [ 0.49436444  0.47457816] 0
#  [ 0.39199018  0.36258182] 0
#  [ 0.56913895  0.40435025] 0
#  [ 0.13878646  0.10810273] 0
#  [ 0.4451918   0.42149922] 0
#  [ 0.42174737  0.27069656] 0
#  [ 0.56559253  0.53261264] 0
#  [ 0.33019338  0.29130903] 0
#  [ 0.49218122  0.34113237] 0
# ]
# [CLASS] =  3 [LEN] =  41686
# [4301 7791 2553 5653  551 6664  142 8091  988 4952]
# [[ 0.33621835  0.32404601] 0
#  [ 0.08542021  0.08035013] 2
#  [ 0.41458059  0.39896416] 0
#  [ 0.20174773  0.18880978] 1
#  [ 0.55305073  0.22284996] 0
#  [ 0.14116073  0.1323824 ] 2 ?
#  [ 0.91265131  0.18499277] 0
#  [ 0.03418902  0.03022966] 2
#  [ 0.3531528   0.18779236] 1
#  [ 0.26382295  0.25655013] 1
# ]
# [CLASS] =  4 [LEN] =  24488
# [2297 4687 1086  655 1421 5546 3853 1697  293 2953]
# [[ 0.09706504  0.09706504] 2
#  [ 0.02883949  0.0288377 ] 2
#  [ 0.19661469  0.19629132] 1
#  [ 0.23908536  0.23900189] 1 ?
#  [ 0.15959639  0.15921254] 1
#  [ 0.00962876  0.009617  ] 2
#  [ 0.04911947  0.04908778] 2
#  [ 0.12655435  0.1264952 ] 1 ?
#  [ 0.29190703  0.29076996] 0
#  [ 0.07148955  0.07144752] 2
# ]
# [CLASS] =  5 [LEN] =  32244
# [3598 4035 3997  583 1850 4016 3431 1498 4162 5074]
# [[ 0.23283065  0.16639155] 1
#  [ 0.48918621  0.3063153 ] 0
#  [ 0.32239569  0.26983345] 0
#  [ 0.77822772  0.29458302] 0
#  [ 0.6076687   0.31834713] 0
#  [ 0.41881219  0.24365171] 0 ?
#  [ 0.32766118  0.19072375] 1
#  [ 0.12847214  0.08919369] 1
#  [ 0.47803635  0.40020559] 0
#  [ 0.40258306  0.33674317] 0
# ]
# [CLASS] =  6 [LEN] =  22082
# [1375 2714 2399 2970 1120 1640 3201 2237 1852 2574]
# [[ 0.57557487  0.28561724] 1 ?
#  [ 0.28949652  0.22279633] 1
#  [ 0.21968493  0.13039943] 1
#  [ 0.46736576  0.26939574] 1 ?
#  [ 0.1036961   0.05850331] 1
#  [ 0.49028152  0.15951022] 1
#  [ 0.39117208  0.2110807 ] 1
#  [ 0.33497338  0.12948272] 1
#  [ 0.47970444  0.38011452] 1
#  [ 0.37444845  0.31052684] 1
# ]
# [CLASS] =  7 [LEN] =  55993
# [5855 9530 5657 7616 3753 2496 7759  308 8246 4773]
# [[ 0.20028194  0.16010473] 1
#  [ 0.3866023   0.36841283] 0
#  [ 0.5011829   0.33514817] 0
#  [ 0.27361221  0.22718388] 1
#  [ 0.11343259  0.08668478] 2
#  [ 0.62205755  0.29670241] 0
#  [ 0.44945837  0.43220667] 0
#  [ 0.88123008  0.2566732 ] 0
#  [ 0.32051852  0.30535357] 1
#  [ 0.40844269  0.25276854] 1
# ]
# [CLASS] =  8 [LEN] =  63297
# [ 9541 10103  4642  2237 11258  7175  4405  2098   750 11088]
# [[ 0.37096522  0.36360553] 0
#  [ 0.45728666  0.45184567] 0
#  [ 0.29686529  0.28760229] 1 ？
#  [ 0.24233173  0.23310983] 1
#  [ 0.42715901  0.42413739] 0
#  [ 0.3378954   0.3288227 ] 0
#  [ 0.49308639  0.48626919] 0
#  [ 0.46916348  0.37155272] 0
#  [ 0.15856831  0.1511377 ] 1
#  [ 0.39911694  0.39516621] 0
# ]
# [CLASS] =  9 [LEN] =  27039
# [3129 4416   95 2862 5001 4153 2159  330  772 4122]
# [[ 0.37824408  0.23253686] 1
#  [ 0.23254965  0.11153634] 1 ? 2
#  [ 1.33523534  0.19395297] 0
#  [ 0.47097554  0.27592186] 1
#  [ 0.15397433  0.07349463] 1 ? 2
#  [ 0.30541755  0.16535682] 1
#  [ 0.42379246  0.13985381] 1
#  [ 0.91716496  0.20752727] 0
#  [ 0.62162343  0.20557869] 0
#  [ 0.07853451  0.03793625] 2
# ]
