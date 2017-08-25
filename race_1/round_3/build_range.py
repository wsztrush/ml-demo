import numpy as np
import os
import time
import multiprocessing
import matplotlib
import race_util
import build_classifier

from sklearn.externals import joblib
from matplotlib import pyplot as plt
from obspy import read

clf = joblib.load(race_util.MODEL_FILE)



def split_range(file_std, file_std_mean, left, right, std_mean_limit):
    file_std = file_std[left:right]
    file_std_mean = file_std_mean[int(left / 5):int(right / 5)]

    index = np.where(file_std_mean > std_mean_limit)[0]
    tmp = index[1:] - index[:-1] - 2
    tmp = np.where(tmp >= 0)[0]

    last_index = 0
    result = []

    for i in tmp:
        if i - last_index > 15:
            l, r = index[last_index + 1] * 5, index[i - 1] * 5
            std_max = np.max(file_std[l:r])
            if std_max > 500:
                result.append((left + l, left + r))

        last_index = i

    return result


def pre_process(unit):
    start = time.time()

    file_std = np.load(race_util.STD_PATH + unit)
    file_std = np.sqrt(np.square(file_std[0]) + np.square(file_std[1]))
    file_std_mean = np.mean(file_std.reshape(-1, 5), axis=1)

    tmp = split_range(file_std, file_std_mean, 0, len(file_std), 200)
    result = []
    for left, right in tmp:
        result.append((left, right))

    if len(result) > 0:
        np.save(race_util.PRE_RANGE_PATH + unit, result)

    print(unit, len(result), time.time() - start)


def process(unit):
    start = time.time()

    # 加工
    file_std = np.load(race_util.STD_PATH + unit)
    file_std = np.sqrt(np.square(file_std[0]) + np.square(file_std[1]))
    file_std_mean = np.mean(file_std.reshape(-1, 5), axis=1)
    file_data = read(race_util.DIR_PATH + unit[:-4] + ".BHZ")[0].data

    # 循环切割
    std_mean_limit = 50

    result = []
    tmp = split_range(file_std, file_std_mean, 0, len(file_std), std_mean_limit)
    while len(tmp) > 0 and std_mean_limit < 300:
        std_mean_limit *= 1.2
        print('std_mean_limit = ', std_mean_limit, len(tmp))
        next_tmp = []

        for left, right in tmp:
            new_left = max(int(left - (right - left) * 0.2), 0)

            x = build_classifier.get_feature(file_std, left, right)

            if clf.predict([x]) == 1:
                plt.subplot(2, 1, 1)
                plt.axvline(x=left - new_left, color='r')
                plt.plot(np.arange(right - new_left), file_std[new_left:right])

                plt.subplot(2, 1, 2)
                plt.axvline(x=int((left - new_left) * 5), color='r')
                print(left, new_left, right)
                plt.plot(np.arange(right * 5 - new_left * 5), file_data[new_left * 5:right * 5])
                plt.show()
                result.append((left, right))

                # if np.max(file_std[left:right] > 10000):
                #     next_tmp += [(int(left + (right - left) * 0.3), right)]
            else:
                split_ret = split_range(file_std, file_std_mean, left, right, std_mean_limit)

                plt.subplot(2, 1, 1)
                plt.axvline(x=left - new_left, color='r')
                plt.plot(np.arange(right - new_left), file_std[new_left:right], color='red')

                plt.subplot(2, 1, 2)
                plt.axvline(x=int((left - new_left) * 5), color='r')
                plt.plot(np.arange(right * 5 - new_left * 5), file_data[new_left * 5:right * 5], color='red')
                plt.show()

                result.append((left, right))

                next_tmp += split_ret

        tmp = next_tmp

    # if len(result) > 0:
    #     np.save(race_util.RANGE_PATH + unit, result)

    print(unit, len(result), time.time() - start)


def main():
    # unit_set = set()
    # for unit in os.listdir(race_util.STD_PATH):
    #     unit_set.add(unit)
    #
    # pool = multiprocessing.Pool(processes=4)
    # pool.map(pre_process, unit_set)

    process('XX.PWU.2008211000000.npy')


if __name__ == '__main__':
    race_util.config()
    main()
