import numpy as np
import os
import time
import multiprocessing
import race_util
import random

from matplotlib import pyplot as plt
from obspy import read


def process(unit, is_show=False):
    start_time = time.time()

    shock_value = np.load('./data/shock/' + unit)
    shock_z_value = np.load('./data/shock_z/' + unit)
    range_value = np.load('./data/all_range/' + unit)
    origin_value_n = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data
    origin_value_z = read(race_util.origin_dir_path + unit[:-4] + '.BHZ')[0].data

    result = []
    for left, right in range_value:
        if race_util.range_filter(shock_value, shock_z_value, left, right):
            result.append([left, right])

            def show():
                before_left = max(int(left - 100), 0)

                plt.subplot(3, 1, 1)
                plt.axhline(y=np.mean(shock_value[left - 10:left]), linestyle='-.', color='red')
                plt.axhline(y=0, color='black', linestyle='-.')
                for i in np.arange(8):
                    plt.axvline(x=left - before_left + (right - left) * (i / 8), color='green', linestyle='-.')
                if race_util.debug_jump_point is not None:
                    for i in race_util.debug_jump_point:
                        plt.axvline(x=left - before_left + i, linestyle='-.', color="red")
                plt.plot(np.arange(right - before_left), shock_value[before_left:right])

                plt.subplot(3, 1, 2)
                origin_left, origin_right = before_left * race_util.step, right * race_util.step
                plt.axvline(x=(left - before_left) * race_util.step, linestyle='-.')
                plt.plot(np.arange(origin_right - origin_left), origin_value_n[origin_left:origin_right])

                plt.subplot(3, 1, 3)
                origin_left, origin_right = before_left * race_util.step, right * race_util.step
                plt.axvline(x=(left - before_left) * race_util.step, linestyle='-.')
                plt.plot(np.arange(origin_right - origin_left), origin_value_z[origin_left:origin_right])

                plt.show()

            if is_show:
                show()

    if len(result) > 0:
        np.save('./data/range/' + unit, result)

    print('[COST]', unit, len(result), time.time() - start_time)


def mult_process():
    pool = multiprocessing.Pool(processes=4)
    pool.map(process, os.listdir('./data/all_range/'))

    total = 0
    for unit in os.listdir('./data/range/'):
        range_list = np.load('./data/range/' + unit)

        total += len(range_list)

    print('[TOTAL RANGE]', total)


def order_process():
    unit_list = os.listdir('./data/all_range/')
    random.shuffle(unit_list)
    for unit in unit_list:
        print(unit)
        process(unit, True)


def check():
    r_1 = []
    r_2 = []

    for unit in os.listdir('./data/all_range/'):
        print(unit)
        shock_value = np.load('./data/shock/' + unit)
        shock_z_value = np.load('./data/shock_z/' + unit)
        range_value = np.load('./data/all_range/' + unit)

        for left, right in range_value:
            filter_ret = race_util.range_filter(shock_value, shock_z_value, left, right)

            if filter_ret:
                r_1.append(np.mean(shock_z_value[left - 20:left]) / np.mean(shock_z_value[left:left + 20]))

    print(np.histogram(r_1, bins=10, range=(0, 1)))
    plt.hist(r_1, bins=10, range=(0, 1))
    plt.show()

    # print(np.histogram(b, bins=10, range=(0, 0.1)))
    # plt.hist(b, bins=10, range=(0, 1))
    # plt.show()


if __name__ == '__main__':
    # race_util.config()
    mult_process()
    # order_process()
    # process('XX.YZP.2008195000000.npy', True)
    # check()

# (array([ 418, 1286,  812,  204,   64,   15,    7,    2,    1,    0]), array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]))
