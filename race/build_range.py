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
    # origin_value_n = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data
    # origin_value_e = read(race_util.origin_dir_path + unit[:-4] + '.BHE')[0].data
    # origin_value_z = read(race_util.origin_dir_path + unit[:-4] + '.BHZ')[0].data

    result = []
    for left, right in range_value:
        filter_ret = race_util.range_filter(shock_value, left, right)

        if filter_ret[0]:
            result.append([left, right])

            # before_left = max(int(left - (right - left) * 0.1), 0)
            #
            # plt.subplot(3, 1, 1)
            # plt.axhline(y=np.mean(shock_value[left - 10:left]), linestyle='-.')
            # plt.axhline(y=np.max(shock_value[left:right]) * 0.05, color="red", linestyle='-.')
            # plt.axhline(y=0, color='black', linestyle='-.')
            # plt.axvline(x=left - before_left, color='black', linestyle='-.')
            # for i in filter_ret[1]:
            #     plt.axvline(x=left - before_left + i, color='red', linestyle='-.')
            # plt.plot(np.arange(right - before_left), shock_value[before_left:right])
            #
            # plt.subplot(3, 1, 2)
            # origin_left, origin_right = before_left * race_util.step, right * race_util.step
            # plt.axvline(x=(left - before_left) * race_util.step, linestyle='-.')
            # plt.plot(np.arange(origin_right - origin_left), origin_value_n[origin_left:origin_right])
            #
            # plt.subplot(3, 1, 3)
            # origin_left, origin_right = before_left * race_util.step, right * race_util.step
            # plt.axvline(x=(left - before_left) * race_util.step, linestyle='-.')
            # plt.plot(np.arange(origin_right - origin_left), origin_value_z[origin_left:origin_right])
            #
            # plt.show()

    if len(result) > 0:
        np.save('./data/range/' + unit, result)

    print('[COST]', unit, len(result), time.time() - start_time)


def main():
    pool = multiprocessing.Pool(processes=4)
    pool.map(process, os.listdir('./data/all_range/'))

    total = 0
    for unit in os.listdir('./data/range/'):
        range_list = np.load('./data/range/' + unit)

        total += len(range_list)

    print('[TOTAL RANGE]', total)

    # unit_list = os.listdir('./data/all_range/')
    # random.shuffle(unit_list)
    # for unit in unit_list:
    #     print(unit)
    #     process(unit)
    #
    # process("XX.YZP.2008200000000.npy")


def check():
    ret = []
    for unit in os.listdir('./data/all_range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        range_value = np.load('./data/all_range/' + unit)

        for left, right in range_value:
            filter_ret = race_util.range_filter(shock_value, left, right)

            if filter_ret[0]:
                ret.append(np.mean(shock_value[left - 20:left]) / (np.mean(shock_value[left:left + 20]) + 1))

    print(np.histogram(ret, bins=10, range=(0, 1)))
    plt.hist(ret, bins=10, range=(0, 1))
    plt.show()


if __name__ == '__main__':
    race_util.config()

    main()
    # check()
