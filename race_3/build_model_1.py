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


def build_before_ratio(shock_value, left, right):
    a = max(left - 10, 0)
    b = int(max((right - left) / 9, 10))

    a = np.mean(shock_value[a:left])
    b = np.mean(shock_value[left:left + b])

    return a / b, a, b


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
                before_ratio = build_before_ratio(shock_value, left, right)

                if feature is None:
                    continue
                predict_ret = kmeans.predict([feature])[0]

                if predict_ret == 9 and 0.5 > before_ratio[0] > 0.4:
                    total += 1
                    print(total, before_ratio[1])
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


def check():
    model_1 = joblib.load('./data/model_1')

    x_dict = dict()
    for unit in os.listdir('./data/all_range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_list = np.load('./data/all_range/' + unit)

        for left, right in range_list:
            feature_1 = build_feature(shock_value, left, right)

            if feature_1 is None:
                continue

            c = model_1.predict([feature_1])[0]
            l = x_dict.get(c)
            if l is None:
                l = x_dict[c] = []

            a = max(left - 10, 0)
            b = int(max((right - left) / 9, 10))

            l.append(np.mean(shock_value[a:left]) / np.mean(shock_value[left:left + b]))

    for i in np.arange(10):
        print(i, np.histogram(x_dict[i], bins=12, range=(0, 0.6)))
        plt.hist(x_dict[i], bins=12, range=(0, 0.6))
        plt.show()


if __name__ == '__main__':
    # race_util.config()

    train()
    # view()
    # check()

# [TOTAL] 203421
# [29724 19024 14871 25068 13667 24776 21253 15160 19610 20268]
# [[ 0.79329404  0.95820825  0.75206183  0.5484478   0.44273829  0.38159168
#    0.33827761  0.27828292]
#  [ 0.9599896   0.83516149  0.45048293  0.3019599   0.23237167  0.1972996
#    0.17459106  0.14552954]
#  [ 0.43878149  0.36137895  0.46121559  0.99407123  0.59194551  0.36137432
#    0.26542609  0.20100425]
#  [ 0.48998824  0.99946804  0.51070729  0.30603341  0.22643527  0.18443684
#    0.15637567  0.12707162]
#  [ 0.51813366  0.41651169  0.37276548  0.4784575   0.87297991  0.70210198
#    0.4473571   0.29138023]
#  [ 0.82292401  0.71482907  0.70835893  0.72791213  0.77414962  0.8022267
#    0.78354168  0.59354106]
#  [ 0.43986391  0.50176105  0.99814058  0.56286403  0.33852921  0.24703118
#    0.19850576  0.15792184]
#  [ 0.99920339  0.38684621  0.20030548  0.14573146  0.12022549  0.10360725
#    0.08855713  0.06967785]
#  [ 0.7404439   0.71092135  0.86486155  0.86000059  0.67306016  0.51495862
#    0.42334112  0.33126559]
#  [ 0.9777741   0.66013775  0.4944737   0.48430825  0.48486116  0.48847504
#    0.47620684  0.34715095]]
