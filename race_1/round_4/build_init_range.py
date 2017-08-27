import numpy as np
import os
import multiprocessing
import race_util
import random

from matplotlib import pyplot as plt


def process(unit):
    shock_value = race_util.load_shock_value(unit)

    result = race_util.split_range(shock_value, 0, len(shock_value), shock_limit=100)

    if len(result) > 0:
        np.save(race_util.init_range_path + unit, result)


def main():
    unit_list = os.listdir(race_util.shock_path)
    random.shuffle(unit_list)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_list)


def check():
    total = 0
    unit_list = os.listdir(race_util.init_range_path)
    for unit in unit_list:
        shock_value = race_util.load_shock_value(unit)
        index = np.where(shock_value == 0)[0]

        total += len(index)

    print(total)



    # total_range = 0
    # total_len = 0
    #
    # unit_list = os.listdir(race_util.init_range_path)
    # for unit in unit_list:
    #     r = np.load(race_util.init_range_path + unit)
    #
    #     total_range += len(r)
    #
    #     for left, right in r:
    #         total_len += right - left
    #
    # print(total_range, total_len, total_len / (864000 * len(unit_list)))


def view():
    unit_list = os.listdir(race_util.init_range_path)
    random.shuffle(unit_list)

    for unit in unit_list:
        range_list = np.load(race_util.init_range_path + unit)
        shock_value = race_util.load_shock_value(unit)

        for left, right in range_list:
            before_left = race_util.get_before_left(left, right)
            plt.axvline(x=left - before_left)
            plt.plot(np.arange(right - before_left), shock_value[before_left:right])
            plt.show()


if __name__ == '__main__':
    race_util.config()

    # main()
    # check()
    view()
