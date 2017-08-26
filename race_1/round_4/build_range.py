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
    if last_index > 0:
        l, r = index[last_index] * race_util.stock_mean_step + left, len(b) * race_util.stock_mean_step + left
        if (r - l) * race_util.stock_step > 500 and np.max(stock_value[l:r] > 1000):
            result.append((l, r))

    return result


def process(unit):
    start = time.time()

    # 加载模型
    model_1 = joblib.load(race_util.model_1)

    # 加载数据
    stock_value = np.load(race_util.stock_path + unit)[0]
    stock_mean_value = np.mean(stock_value.reshape(-1, race_util.stock_mean_step), axis=1)

    # 分割
    tmp = split_range(stock_value, stock_mean_value, 0, len(stock_value), 50)
    result = []
    for left, right in tmp:
        f = build_model_1.get_feature(stock_value, left, right)
        if f and model_1.predict([f])[0] == 1:
            result.append((left, right))

    # 展示
    # file_data = read(race_util.origin_dir_path + unit[:-4] + ".BHZ")[0].data
    # for left, right in result:
    #     f = build_model_1.get_feature(stock_value, left, right)
    #
    #     if not f:
    #         continue
    #
    #     print(left, right, model_1.predict([f])[0])
    #
    #     c = 'green'
    #     if model_1.predict([f])[0] == 0:
    #         c = 'red'
    #
    #     before_left = race_util.get_before_left(left, right)
    #
    #     plt.subplot(2, 1, 1)
    #     plt.axvline(x=left - before_left, color='r')
    #     plt.bar(np.arange(right - before_left), stock_value[before_left:right])
    #
    #     plt.subplot(2, 1, 2)
    #     plt.axvline(x=int((left - before_left) * race_util.stock_step), color='r')
    #     plt.plot(np.arange(right * race_util.stock_step - before_left * race_util.stock_step), file_data[before_left * race_util.stock_step:right * race_util.stock_step], color=c)
    #
    #     # 模型的数据展示
    #     # x = build_rbm.get_feature(stock_value, left, right)
    #     # t = model.transform([x])[0]
    #     #
    #     # plt.subplot(2, 2, 3)
    #     # plt.bar(np.arange(len(x)), x)
    #     #
    #     # plt.subplot(2, 2, 4)
    #     # plt.ylim(0, 1)
    #     # plt.bar(np.arange(len(t)), t)
    #
    #     plt.show()

    # 保存到文件
    if len(result) > 0:
        np.save(race_util.range_path + unit, result)

    print(unit, len(result), time.time() - start)


def main():
    unit_set = set()
    unit_list = os.listdir(race_util.stock_path)
    random.shuffle(unit_list)

    for unit in unit_list:
        unit_set.add(unit)

    print(len(unit_set))

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)

    # process('XX.YZP.2008213000000.npy')


if __name__ == '__main__':
    race_util.config()

    main()
