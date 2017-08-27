import numpy as np
import os
import multiprocessing
import race_util
import random


def process(unit):
    shock_value_list = np.load(race_util.shock_path + unit)
    shock_value = np.sqrt(np.square(shock_value_list[0]) + np.square(shock_value_list[1]) + np.square(shock_value_list[2]))

    result = race_util.split_range(shock_value, 0, len(shock_value))

    if len(result) > 0:
        np.save(race_util.init_range_path + unit, result)


def main():
    unit_list = os.listdir(race_util.shock_path)
    random.shuffle(unit_list)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_list)


def check():
    total_range = 0
    total_len = 0

    unit_list = os.listdir(race_util.init_range_path)
    for unit in unit_list:
        r = np.load(race_util.init_range_path + unit)

        total_range += len(r)

        for left, right in r:
            total_len += right - left

    print(total_range, total_len, total_len / (864000 * len(unit_list)))


if __name__ == '__main__':
    # main()
    check()
