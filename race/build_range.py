import numpy as np
import os
import time
import multiprocessing
import race_util
import random

from matplotlib import pyplot as plt
from obspy import read


def process(unit):
    start_time = time.time()

    shock_value = np.load('./data/shock/' + unit)
    range_value = np.load('./data/all_range/' + unit)
    origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHE')[0].data

    result = []
    for left, right in range_value:
        filter_ret = race_util.range_filter(shock_value, left, right)

        if filter_ret[0]:
            result.append([left, right])

            before_left = max(int(left - (right - left) * 0.1), 0)

            plt.subplot(2, 1, 1)
            plt.axhline(y=np.mean(shock_value[left - 10:left]), linestyle='-.')
            plt.axhline(y=0, color='black', linestyle='-.')
            plt.axvline(x=left - before_left, color='black', linestyle='-.')
            for i in filter_ret[1]:
                plt.axvline(x=left - before_left + i, color='red', linestyle='-.')
            plt.plot(np.arange(right - before_left), shock_value[before_left:right])

            plt.subplot(2, 1, 2)
            origin_left, origin_right = before_left * race_util.step, right * race_util.step
            plt.axvline(x=(left - before_left) * race_util.step, linestyle='-.')
            plt.plot(np.arange(origin_right - origin_left), origin_value[origin_left:origin_right])

            plt.show()

    if len(result) > 0:
        np.save('./data/range/' + unit, result)

    print('[COST]', unit, len(result), time.time() - start_time)


def main():
    # pool = multiprocessing.Pool(processes=4)
    # pool.map(process, os.listdir('./data/all_range/'))
    #
    # total = 0
    # for unit in os.listdir('./data/range/'):
    #     range_list = np.load('./data/range/' + unit)
    #
    #     total += len(range_list)
    #
    # print('[TOTAL RANGE]', total)

    # unit_list = os.listdir('./data/all_range/')
    # random.shuffle(unit_list)
    # for unit in unit_list:
    #     print(unit)
    #     process(unit)

    process("XX.YZP.2008195000000.npy")


if __name__ == '__main__':
    race_util.config()

    main()
