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

flag = [
    [0, 1], [0, 4],
    [1, 3],
    [2, 0], [2, 1], [2, 4],
    [4, 1], [4, 4],
    [5, 1], [5, 3],
    [6, 0], [6, 2], [6, 4],
    [7, 0], [7, 1], [7, 2], [7, 4],
    [8, 1], [9, 2], [9, 4]
]


def build_feature(shock_value, left, right):
    before_left = max(left - 10, int(left - (right - left) / 9), 0)
    if before_left == left:
        return None

    right -= (right - left) % 9
    if right - left < 9:
        return None

    a = np.mean(shock_value[before_left:left])
    b = np.mean(shock_value[left:right].reshape(9, -1), axis=1) + 1.0
    b_max = np.max(b)

    return [
        a / b[0],
        a / b_max,
    ]


def train():
    x_dict = dict()
    for unit in os.listdir('./data/range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_list = np.load('./data/range/' + unit)

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

    for i in x_dict:
        print('[CLASS] = ', i, '[LEN] = ', len(x_dict[i]))
        kmeans = KMeans(n_clusters=5).fit(x_dict[i])

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

                feature_1 = build_model_1.build_feature(shock_value, left, right)
                feature_2 = build_feature(shock_value, left, right)

                if feature_1 is None or feature_2 is None:
                    continue

                predict_1 = model_1.predict([feature_1])[0]
                predict_2 = model_2[predict_1].predict([feature_2])[0]

                if predict_1 == 9 and predict_2 == 0:
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

    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=200, repeat=False)
    plt.show()


if __name__ == '__main__':
    race_util.config()

    # train()
    view()

# [CLASS] =  0 [LEN] =  31090
# [5809 8663 4926 5214 6478]
# [[ 0.98258699  0.17636235]
#  [ 0.10018933  0.03505571] * [0,1]
#  [ 0.52308155  0.12257624] * [0,2] ?
#  [ 0.77106739  0.14579118] * [0,3] ?
#  [ 0.28917098  0.08770233] * [0,4]
# ]
# ============================
# [CLASS] =  1 [LEN] =  41589
# [10941  6321 12026  3892  8409]
# [[ 1.01150677  0.31343939]
#  [ 0.43982813  0.16748407]
#  [ 0.85298517  0.2612344 ]
#  [ 0.18822108  0.08325375] * [1,3]
#  [ 0.66482783  0.22032354]
# ]
# ============================
# [CLASS] =  2 [LEN] =  27832
# [ 3862 11279  2557  2925  7209]
# [[ 0.40211702  0.12687906] * [2,0] ?
#  [ 0.0663536   0.03008463] * [2,1]
#  [ 0.9378228   0.15307282]
#  [ 0.65402145  0.14756684]
#  [ 0.20770484  0.08234985] * [2,4]
# ]
# ============================
# [CLASS] =  3 [LEN] =  22910
# [5289 4088 5868 1587 6078]
# [[ 1.00091575  0.3422347 ]
#  [ 0.40205714  0.18701464]
#  [ 0.6158947   0.25710931]
#  [ 0.08542374  0.04267897]
#  [ 0.81436386  0.29045769]
# ]
# ============================
# [CLASS] =  4 [LEN] =  21091
# [4094 4891 5054 3878 3174]
# [[ 0.77219222  0.20860177]
#  [ 0.35449814  0.1312734 ] * [4,1]
#  [ 0.55756377  0.18035519]
#  [ 0.99166797  0.25144701]
#  [ 0.13477204  0.05536514] * [4,4]
# ]
# ============================
# [CLASS] =  5 [LEN] =  41691
# [12019  6747  9472  3935  9518]
# [[ 0.81389317  0.30722739]
#  [ 0.41483817  0.21124919] * [5,1]
#  [ 0.62844781  0.26712923]
#  [ 0.16817365  0.10261241] * [5,3]
#  [ 0.98517708  0.36346678]
# ]
# ============================
# [CLASS] =  6 [LEN] =  30372
# [5340 6151 6869 5525 6487]
# [[ 0.5477599   0.13557531] * [6,0]
#  [ 0.9968591   0.214228  ]
#  [ 0.1280278   0.04344706] * [6,2]
#  [ 0.78745827  0.17337532]
#  [ 0.32061498  0.0913026 ] * [6,4]
# ]
# ============================
# [CLASS] =  7 [LEN] =  8876
# [ 248 2132  785   24 5687]
# [[ 0.2343957   0.22611526] * [7,0]
#  [ 0.05452628  0.05136303] * [7,1]
#  [ 0.1231528   0.11851601] * [7,2]
#  [ 0.82906099  0.65531492]
#  [ 0.01219551  0.0116312 ] * [7,4]
# ]
# ============================
# [CLASS] =  8 [LEN] =  54171
# [13191  2816 13609  7520 17035]
# [[ 0.68584442  0.30352086]
#  [ 0.19630731  0.10295534] * [8,1]
#  [ 1.00456532  0.40465399]
#  [ 0.4831819   0.23409718]
#  [ 0.84828994  0.35215109]
# ]
# ============================
# [CLASS] =  9 [LEN] =  27154
# [5942 5420 4672 4940 6180]
# [[ 0.53664145  0.15096187] * [9,0] ?
#  [ 0.99245055  0.23441007]
#  [ 0.13474053  0.04961663] * [9,2]
#  [ 0.76607255  0.19043372]
#  [ 0.3348755   0.10868325] * [9,4]
# ]
# ============================
