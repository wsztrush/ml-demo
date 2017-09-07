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
            jump_index = np.load('./data/jump/' + unit)
            origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

            for left, right in range_list:
                before_left = max(int(left - (right - left) / 9), 0)

                tmp = shock_value[left:right]
                max_index = np.where(tmp == np.max(tmp))

                inner_jump_index = np.where((jump_index > left) & (jump_index < right))[0]

                feature = build_feature(shock_value, left, right)
                # before_ratio = build_before_ratio(shock_value, left, right)

                if feature is None:
                    continue
                predict_ret = kmeans.predict([feature])[0]

                if predict_ret == 0:
                    total += 1
                    print(total, max_index - left, inner_jump_index)
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
    # train()
    view()
    # check()


    # [TOTAL] 204098
    # [22693 19247 14650 20245 25476 19699 13982 11062 26090 30954]
    # [[ 0.77179115  0.83912615  0.87572828  0.78475361  0.62575825  0.49493578
    #    0.40337117  0.31142242] - 0
    #  [ 0.4019137   0.45556173  0.99887221  0.5524088   0.32652006  0.23626844
    #    0.18746132  0.14689778] - 1
    #  [ 0.99897634  0.39136644  0.20522641  0.15254674  0.1274821   0.1107679
    #    0.09544468  0.07383921] - 2 【前面有最大值，下降很快】
    #  [ 0.82052546  0.71635254  0.69974519  0.71888237  0.75515737  0.78529059
    #    0.76873116  0.57049291] - 3 【不靠谱的节点】
    #  [ 0.41240933  0.99977357  0.45104685  0.2712625   0.20620832  0.17134652
    #    0.14637     0.11799013] - 4
    #  [ 0.95672441  0.82332095  0.45326183  0.30691824  0.23984868  0.20524504
    #    0.18227777  0.1506807 ] - 5
    #  [ 0.43272446  0.37297696  0.49680569  0.99401494  0.57928403  0.35182082
    #    0.25582217  0.19145805] - 6
    #  [ 0.5245439   0.43181387  0.37530369  0.47298749  0.85961657  0.68740155
    #    0.43101763  0.27060816] - 7
    #  [ 0.57567491  0.979649    0.75387853  0.48917593  0.37174906  0.30782975
    #    0.26305608  0.21024995] - 8
    #  [ 0.96541874  0.8033538   0.57998867  0.50351136  0.45987328  0.42646528
    #    0.39547216  0.30984713] - 9
    # ]
