import numpy as np
import race_util
import os
import random

from matplotlib import pyplot as plt
from obspy import read


def process(unit):
    stock_value = np.load(race_util.stock_path + unit)[0]
    range_value = np.load(race_util.range_path + unit)
    file_data = read(race_util.origin_dir_path + unit[:-4] + ".BHE")[0].data

    print('total', len(range_value))

    plot_count = 5
    for i in np.arange(0, len(range_value), plot_count):
        if i % 20 == 0:
            print('-----')
        print(int(i / 5))

        for j in np.arange(min(len(range_value) - i, plot_count)):
            lr = range_value[i + j]
            left, right = lr[0], lr[1]
            new_left = max(int(left - (right - left) * 0.1), 0)

            plt.subplot(plot_count, 2, 1 + j * 2)
            plt.axvline(x=left - new_left, color='r')
            plt.plot(np.arange(right - new_left), stock_value[new_left:right])

            plt.subplot(plot_count, 2, 2 + j * 2)
            plt.axvline(x=int((left - new_left) * race_util.stock_step), color='r')
            plt.plot(np.arange(right * race_util.stock_step - new_left * race_util.stock_step), file_data[new_left * race_util.stock_step:right * race_util.stock_step])

        plt.show()


def save_sample_1():
    ret = []

    np.save(race_util.sample_1_file, ret)


def save_sample(sample_file):
    # 给数据打标
    flag = dict()

    s = flag['XX.JMG.2008194000000.npy'] = []
    s += [0, 1, 1, 1, 0] + [1, 0, 1, 1, 1] + [0, 0, 0, 1, 1] + [0, 0, 0, 1, 1]
    s += [1, 1, 1, 1, 0] + [0, 0, 0, 1, 1] + [1, 0, 0, 1, 1] + [0, 1, 0, 1, 1]
    s += [0, 0, 1, 1, 1] + [0, 1, 0, 0, 0] + [0, 0, 0, 1, 1] + [0, 1, 1, 0, 0]
    s += [0, 0, 1, 1, 1] + [1, 1, 1, 0, 0] + [1, 0, 1, 0, 1] + [1, 0, 1, 1, 1]
    s += [1, 1, 1, 1, 1] + [0, 1, 1, 1, 1] + [0, 0, 1, 1, 1] + [0, 1, 0, 1, 1]
    s += [0, 0, 1, 1, 1] + [0, 0, 1, 1, 1] + [1, 1, 0, 1, 1] + [0, 0, 1, 1, 1]
    s += [1,1,] + [] + [] + []
    s += [] + [] + [] + []
    s += [] + [] + [] + []
    s += [] + [] + [] + []

    s = flag[''] = []
    s += [] + [] + [] + []

    # 生成并保存范围文件
    ret = dict()
    if os.path.exists(sample_file):
        ret = np.load(sample_file)

    for u, f in ret:
        s = ret.get(u)
        if not s:
            s = ret[u] = []

        range_value = np.load(race_util.range_path + u)

        for i in np.arange(len(f)):
            if f[i] in [0, 1]:
                left, right = range_value[i]
                s.append((left, right, f[i]))

    np.save(race_util.sample_2_file, ret)


if __name__ == '__main__':
    race_util.config()

    process('XX.JMG.2008194000000.npy')

    # save_sample()
    # total = 0
    # unit_list = os.listdir(race_util.range_path)
    # random.shuffle(unit_list)
    # for unit in unit_list:
    #     range_value = np.load(race_util.range_path + unit)
    #     total += len(range_value)
    #     print(unit, len(range_value))
    #
    # print(total)
