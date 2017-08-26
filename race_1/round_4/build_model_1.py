# 使用GBDT来判断左边的分割点是否靠谱。

import numpy as np
import os
import time
import random
import race_util

from obspy import read
from matplotlib import animation
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier


def get_feature(stock_value, left, right):
    move_size = int((right - left) / 20)

    if left > move_size * 2:
        a0 = np.mean(stock_value[left - 2 * move_size:left])
        a1 = np.mean(stock_value[left - move_size:left])
        a2 = np.mean(stock_value[left - 2 * move_size:left - move_size])

        b0 = np.mean(stock_value[left:left + 2 * move_size]) + 1.0
        b1 = np.mean(stock_value[left:left + move_size]) + 1.0
        b2 = np.mean(stock_value[left + move_size:left + 2 * move_size]) + 1.0

        stock_max = np.max(stock_value[left:right]) + 1.0

        return [a0 / b0, a1 / b1, a2 / b2, a0 / stock_max, b0 / stock_max]


def train():
    sample_list = np.load(race_util.sample_1_file)

    x_list = []
    y_list = []
    range_list = []

    for sample in sample_list:
        unit = sample[0]

        stock_value = np.load(race_util.stock_path + unit)[0]
        range_value = np.load(race_util.range_path + unit)

        for i in np.arange(0, len(range_value)):
            if sample[i + 1] == 2:
                continue

            left, right = range_value[i]

            f = get_feature(stock_value, left, right)
            if f:
                x_list.append(f)
                y_list.append(sample[i + 1])

            range_list.append((unit, left, right, sample[i + 1]))

    model = GradientBoostingClassifier()
    model.fit(x_list, y_list)

    joblib.dump(model, race_util.model_1)

    # 打印准确率
    pre_y = model.predict(x_list)
    index = (np.where(pre_y != y_list))[0]
    print(len(index), len(y_list), len(index) / len(y_list))

    # 查看分错的
    for i in index:
        unit, left, right, sign_y = range_list[i]

        print(sign_y)

        stock_value = np.load(race_util.stock_path + unit)[0]
        file_data = read(race_util.origin_dir_path + unit[:-4] + ".BHZ")[0].data

        before_left = race_util.get_before_left(left, right)

        plt.subplot(2, 1, 1)
        plt.axvline(x=left - before_left, color='r')
        plt.plot(np.arange(right - before_left), stock_value[before_left:right])

        plt.subplot(2, 1, 2)
        plt.axvline(x=int((left - before_left) * race_util.stock_step), color='r')
        plt.plot(np.arange(right * race_util.stock_step - before_left * race_util.stock_step), file_data[before_left * race_util.stock_step:right * race_util.stock_step])
        plt.show()


if __name__ == '__main__':
    train()
