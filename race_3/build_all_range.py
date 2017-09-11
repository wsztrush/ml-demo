import numpy as np
import os
import time
import multiprocessing
import race_util
import random

from obspy import read
from matplotlib import pyplot as plt


def process(unit):
    start_time = time.time()

    jump_index = np.load('./data/jump/' + unit)
    shock_value = np.load('./data/shock/' + unit)
    shock_mean_value_list = [np.mean(shock_value[i:i - 10].reshape(-1, 10), axis=1) for i in np.arange(10)]
    # origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

    ret = []
    for index in jump_index:
        start_mean_index = int(index / 10 - 1)

        shock_mean_value = shock_mean_value_list[index % 10]
        v = shock_mean_value[start_mean_index]
        v_list = [v]

        for i in np.arange(1, len(shock_mean_value) - start_mean_index):
            if i % 10 == 0:
                v *= 1.1
            v_list.append(v)

            if shock_mean_value[start_mean_index + i] <= v:
                stop_index = index + i * 10

                if race_util.judge_range(shock_mean_value[start_mean_index], shock_value[index:stop_index]):
                    ret.append([index, stop_index])

                    # 展示
                    # left, right = index, stop_index + 100
                    # before_left = max(left - 100, 0)
                    #
                    # plt.subplot(2, 1, 1)
                    # plt.plot(np.arange(right - before_left), shock_value[before_left:right])
                    # plt.axvline(x=left - before_left, color='black', linestyle='-.')
                    # plt.axhline(y=0, color='black', linestyle='-.')
                    # x = np.arange(left - before_left, left - before_left + i * 10 + 1, 10)
                    # plt.plot(x, v_list, color='black', linestyle=':')
                    # plt.plot(np.arange(left - before_left, left - before_left + i * 10 + 1, 10), shock_mean_value[start_mean_index:start_mean_index + i + 1], color='black', linestyle=':')
                    #
                    # plt.subplot(2, 1, 2)
                    # origin_left, origin_right = before_left * race_util.step, right * race_util.step
                    # plt.plot(np.arange(origin_right - origin_left), origin_value[origin_left:origin_right])
                    #
                    # plt.show()
                break

    if len(ret) > 0:
        np.save('./data/all_range/' + unit, ret)

    print(unit, 'cost=', time.time() - start_time, 'ret=', len(ret))


def main():
    # unit_list = os.listdir('./data/jump/')
    # random.shuffle(unit_list)
    # for unit in unit_list:
    #     process(unit)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, os.listdir('./data/jump/'))

    total = 0
    for unit in os.listdir('./data/all_range/'):
        range_list = np.load('./data/all_range/' + unit)

        total += len(range_list)

    print('total = ', total)  # 229026


if __name__ == '__main__':
    main()
