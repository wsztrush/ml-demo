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
    shock_mean_value = np.mean(shock_value.reshape(-1, 10), axis=1)
    # origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

    result = []

    last_jump_index = 0
    for j_index in jump_index:
        if j_index <= last_jump_index:
            continue

        mean_index = int(j_index / 10)
        mean_limit_value = shock_mean_value[mean_index] + 1.0
        mean_limit_value_list = [mean_limit_value]

        for i in np.arange(1, len(shock_mean_value) - mean_index):
            if i % 10 == 0:
                mean_limit_value *= 1.1
            mean_limit_value_list.append(mean_limit_value)

            if shock_mean_value[mean_index + i] <= mean_limit_value:
                stop_index = j_index + i * 10

                c = 'red'
                if i > 8 and np.max(shock_value[j_index:stop_index]) > 1000:
                    last_jump_index = stop_index
                    result.append((j_index, stop_index))

                    c = 'green'

                # left, right = j_index - 100, stop_index + 100
                # if left > 0:
                #     plt.subplot(2, 1, 1)
                #     plt.axvline(x=100, color='red')
                #     plt.axvline(x=right - left - 100, color='red')
                #     plt.plot(np.arange(right - left), shock_value[left:right], color=c)
                #     plt.plot(np.arange(100, 100 + len(mean_limit_value_list) * 10, 10), shock_mean_value[mean_index:mean_index + i + 1], color='yellow')
                #     plt.plot(np.arange(100, 100 + len(mean_limit_value_list) * 10, 10), mean_limit_value_list, color='green')
                #
                #     plt.subplot(2, 1, 2)
                #     plt.plot(np.arange(right * race_util.shock_step - left * race_util.shock_step), origin_value[left * race_util.shock_step:right * race_util.shock_step])
                #     plt.show()

                break

    if len(result) > 0:
        np.save('./data/range/' + unit, result)

    print('[COST]', unit, len(result), time.time() - start_time)


def main():
    pool = multiprocessing.Pool(processes=4)
    pool.map(process, os.listdir('./data/jump/'))

    # process('XX.JJS.2008186000000.npy')


def check():
    total = 0
    for unit in os.listdir('./data/range/'):
        range_list = np.load('./data/range/' + unit)

        total += len(range_list)

    print(total)


if __name__ == '__main__':
    race_util.config()

    # main()
    check()
