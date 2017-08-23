import numpy as np
import os
import time
import multiprocessing
import matplotlib
import build_clf
import race_config

from sklearn.externals import joblib
from matplotlib import pyplot as plt
from obspy import read

# 加载模型
clf = joblib.load(race_config.MODEL_FILE)


# 切分
def split_range(file_std, file_std_mean, left, right, std_mean_limit):
    file_std = file_std[left:right]
    file_std_mean = file_std_mean[int(left / 5):int(right / 5)]

    index = np.where(file_std_mean > std_mean_limit)[0]
    tmp = index[1:] - index[:-1] - 3
    tmp = np.where(tmp >= 0)[0]

    last_index = 0
    result = []

    for i in tmp:
        if i - last_index > 15:
            l, r = index[last_index + 1] * 5, index[i - 1] * 5
            std_max = np.max(file_std[l:r])
            if std_max > 1000:
                result.append((left + l, left + r))

        last_index = i

    return result


# 初步提取一些特征，用来做模型
def pre_process(unit):
    start = time.time()

    # 加工
    file_std = np.load(race_config.STD_PATH + unit)
    file_std = np.sqrt(np.square(file_std[0]) + np.square(file_std[1]))
    file_std_mean = np.mean(file_std.reshape(-1, 5), axis=1)

    # 切割
    tmp = split_range(file_std, file_std_mean, 0, len(file_std), 100)
    result = []
    for l, r in tmp:
        left, right = max(int(l - (r - l) * 0.1), 0), r
        result.append((left, right))

        # plt.axvline(x=l - left, color='r')
        # plt.plot(np.arange(right - left), file_std[left:right])
        # plt.show()

    # 保存
    if len(result) > 0:
        np.save(race_config.PRE_RANGE_PATH + unit, result)

    print(unit, len(result), time.time() - start)


# 正式递归处理
def process(unit):
    start = time.time()

    # 加工
    file_std = np.load(race_config.STD_PATH + unit)
    file_std = np.sqrt(np.square(file_std[0]) + np.square(file_std[1]))
    file_std_mean = np.mean(file_std.reshape(-1, 5), axis=1)
    file_data = read(race_config.DIR_PATH + unit[:-4] + ".BHZ")

    # 循环切割
    std_mean_limit = 50

    result = []
    tmp = split_range(file_std, file_std_mean, 0, len(file_std), std_mean_limit)
    while len(tmp) > 0 and std_mean_limit < 200:
        std_mean_limit *= 1.2
        print('std_mean_limit = ', std_mean_limit, len(tmp))
        next_tmp = []

        for l, r in tmp:
            left, right = race_config.get_left_right(l, r)

            x = build_clf.get_x(file_std, left, right)

            clf_p = (clf.predict([x]) == 1)
            if clf_p:
                # plt.axvline(x=int((right - left) / 11), color='yellow')
                # plt.plot(np.arange(right - left), file_std[left:right], 'green')
                # plt.show()
                result.append((left, right))

                if np.max(file_std[left:right] > 10000):
                    next_tmp += [(int(left + (right - left) * 0.3), right)]
            else:
                split_ret = split_range(file_std, file_std_mean, l, r, std_mean_limit)
                # plt.axvline(x=int((right - left) / 11), color='yellow')
                # plt.plot(np.arange(right - left), file_std[left:right], 'red')
                # plt.show()

                next_tmp += split_ret

        tmp = next_tmp

    if len(result) > 0:
        np.save(race_config.RANGE_PATH + unit, result)

    print(unit, len(result), time.time() - start)


def main():
    unit_set = set()
    for unit in os.listdir(race_config.STD_PATH):
        unit_set.add(unit)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)

    # process('XX.MXI.2008196000000.npy')


def config():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 8)


if __name__ == '__main__':
    config()
    main()
