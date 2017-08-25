import numpy as np
import os
import time
import multiprocessing
import matplotlib
import race_util

from obspy import read
from matplotlib import pyplot as plt


def split_range(stock_value, stock_mean_value, left, right, stock_mean_limit):
    b = stock_mean_value[int(left / race_util.stock_mean_step): int(right / race_util.stock_mean_step)]

    # 0, 2, 4, 6, 8, 10,12
    # 1, 1, 1, 0, 1, 1, 1
    # -------------------
    # 0, 1, 2, 4, 5, 6
    #    1, 1, 2, 1, 1
    #    0, 0, 1, 0, 0
    # 2, 6
    # 0, 6 | 8, 14
    index = np.where(b > stock_mean_limit)[0]
    continuity_index = np.where(index[1:] - index[:-1] - race_util.stock_mean_gap > 0)[0]

    last_index = 0
    result = []

    # 遍历前面的区间
    for i in continuity_index:
        l, r = index[last_index] * race_util.stock_mean_step + left, (index[i] + 1) * race_util.stock_mean_step + left
        if (r - l) * race_util.stock_step > 500 and np.max(stock_value[l:r] > 1000):
            result.append((l, r))

        last_index = i + 1

    # 处理最后一个区间
    l, r = index[last_index] * race_util.stock_mean_step + left, len(b) * race_util.stock_mean_step + left
    if (r - l) * race_util.stock_step > 500 and np.max(stock_value[l:r] > 1000):
        result.append((l, r))

    return result


def process(unit):
    start = time.time()

    # 加载数据
    stock_value = np.load(race_util.stock_path + unit)[0]
    stock_mean_value = np.mean(stock_value.reshape(-1, race_util.stock_mean_step), axis=1)
    file_data = read(race_util.origin_dir_path + unit[:-4] + ".BHZ")[0].data

    print(len(stock_value), len(stock_mean_value), len(file_data))

    # 分割
    result = split_range(stock_value, stock_mean_value, 0, len(stock_value), 100)

    for left, right in result:
        before_left = race_util.get_before_left(left, right)

        plt.subplot(2, 1, 1)
        plt.axvline(x=left - before_left, color='r')
        plt.bar(np.arange(right - before_left), stock_value[before_left:right])

        plt.subplot(2, 1, 2)
        plt.axvline(x=int((left - before_left) * race_util.stock_step), color='r')

        plt.plot(np.arange(right * race_util.stock_step - before_left * race_util.stock_step), file_data[before_left * race_util.stock_step:right * race_util.stock_step])
        plt.show()

    # 保存到文件
    print(unit, len(result), time.time() - start)


def main():
    process('XX.YZP.2008213000000.npy')


if __name__ == '__main__':
    race_util.config()

    main()
