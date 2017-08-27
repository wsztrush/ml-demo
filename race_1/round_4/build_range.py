import numpy as np
import os
import time
import multiprocessing
import matplotlib
import race_util
import build_rbm
import build_model_1
import random

from obspy import read
from matplotlib import pyplot as plt
from sklearn.externals import joblib

# 加载模型
# model_1 = joblib.load(race_util.model_1)


def split_range(stock_value, left, right, stock_limit):
    b = stock_value[left:right]

    # 0, 2, 4, 6, 8, 10,12
    # 1, 1, 1, 0, 1, 1, 1
    # -------------------
    # 0, 1, 2, 4, 5, 6
    #    1, 1, 2, 1, 1
    #    0, 0, 1, 0, 0
    # 2, 6
    # 0, 6 | 8, 14
    index = np.where(b > stock_limit)[0]
    continuity_index = np.where(index[1:] - index[:-1] - race_util.stock_gap > 0)[0]

    last_index = 0
    result = []

    # 遍历前面的区间
    for i in continuity_index:
        l, r = index[last_index] + left, index[i] + left + 1
        if is_good_lr(stock_value, l, r):
            result.append((l, r))

        last_index = i + 1

    # 处理最后一个区间
    if last_index > 0:
        l, r = index[last_index] + left, len(b) + left
        if is_good_lr(stock_value, l, r):
            result.append((l, r))

    return result


def is_left(stock_value, left, right):
    if not is_good_lr(stock_value, left, right):
        return False

    f = build_model_1.get_feature(stock_value, left, right)
    if f and model_1.predict([f])[0] == 1:
        return True
    return False


def is_good_lr(stock_value, left, right):
    return (right - left) * race_util.stock_step > 500 and np.max(stock_value[left:right] > 1000)


def find_first_left(stock_value, left, right):
    for i in np.arange(left + 1, int(right - 500 / race_util.stock_step), step=max(5, int((right - left) * 0.01))):
        if is_left(stock_value, i, right):
            return i


def plt_range(stock_value, file_data, left, right, c, level):
    before_left = race_util.get_before_left(left, right)

    print('[LEVEL] ', left, right, level)

    plt.subplot(2, 1, 1)
    plt.axvline(x=left - before_left, color='r')
    plt.bar(np.arange(right - before_left), stock_value[before_left:right])

    plt.subplot(2, 1, 2)
    plt.axvline(x=int((left - before_left) * race_util.stock_step), color='r')
    plt.plot(np.arange(right * race_util.stock_step - before_left * race_util.stock_step), file_data[before_left * race_util.stock_step:right * race_util.stock_step], color=c)

    plt.show()


def process(unit):
    start = time.time()

    # 加载数据
    stock_value = np.load(race_util.stock_path + unit)[0]

    # 分割
    # result = split_range(stock_value, 0, len(stock_value), 100)

    # 保存数据
    # result = []
    # for left, right in tmp:
    #     f = build_model_1.get_feature(stock_value, left, right)
    #     if f and model_1.predict([f])[0] == 1:
    #         result.append((left, right))

    # 展示和搜索
    # file_data = read(race_util.origin_dir_path + unit[:-4] + ".BHZ")[0].data

    result = []
    split_limit = 100
    result = split_range(stock_value, 0, len(stock_value), split_limit)
    # while len(tmp) > 0 and split_limit < 1000:
    #     next_tmp = []
    #     split_limit *= 1.5
    #
    #     for left, right in tmp:
    #         if is_left(stock_value, left, right):
    #             result.append((left, right))
    #         else:
    #             new_left = find_first_left(stock_value, left, right)
    #             if new_left:
    #                 result.append((new_left, right))
    #
    #     tmp = next_tmp

    # 保存到文件
    if len(result) > 0:
        np.save(race_util.range_path + unit, result)

    print(unit, len(result), time.time() - start)


def main():
    unit_set = []
    unit_list = os.listdir(race_util.stock_path)
    random.shuffle(unit_list)

    for unit in unit_list:
        unit_set.append(unit)

    print(len(unit_set))

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)

    # for unit in unit_set:
    #     process(unit)

    # process('GS.WXT.2008188000000.npy')


if __name__ == '__main__':
    race_util.config()

    main()
