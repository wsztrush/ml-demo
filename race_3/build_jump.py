import numpy as np
import os
import multiprocessing
import race_util
import time

from matplotlib import pyplot as plt
from obspy import read


def find_jump_point(shock_value, step):
    s = shock_value[step:step - 10]
    s_mean = np.mean(s.reshape(-1, 10), axis=1) + 1
    s_min = np.min(s.reshape(-1, 10), axis=1)

    a = (s_mean[:-1] / s_mean[1:])[:-1]
    b = (s_mean[:-2] / s_mean[2:])

    ret = np.max((a, b), axis=0)
    ret[np.where(s_min[:-2] == 0)] = 1

    return ret


def process(unit):
    start_time = time.time()

    shock_value = np.load('./data/shock/' + unit)

    r = []
    for step in np.arange(10):
        r.append(find_jump_point(shock_value, step))
    r_min = np.min(r, axis=0)

    a = np.zeros(len(r_min), dtype=int)
    for step in np.arange(10):
        a[np.where(r[step] == r_min)] = step
    b = np.arange(0, 10 * len(a), 10, dtype=int)

    ret = (a + b)[np.where(r_min < 0.5)] + 10

    if len(ret) > 0:
        np.save('./data/jump/' + unit, ret)

    print('cost', unit, time.time() - start_time, 'ret', len(ret))


def view(unit):
    jump_index = np.load('./data/jump/' + unit)
    shock_value = np.load('./data/shock/' + unit)
    origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

    for i in jump_index:
        left, right = i - 100, i + 500

        if left < 0:
            continue

        x = np.arange(100, 600, 10)
        y_1 = np.zeros(50)
        y_2 = np.zeros(50)
        y_1[0] = y_2[0] = np.mean(shock_value[i - 10:i])

        for j in np.arange(1, 50):
            y_1[j] = y_1[j - 1] * 1.01
            y_2[j] = y_2[j - 1] * 1.02

        tmp = np.mean(shock_value[left:right].reshape(-1, 10), axis=1)
        plt.plot(np.arange(0, right - left, 10), tmp)

        plt.subplot(2, 1, 1)
        plt.plot(x, y_1, linestyle=':')
        plt.plot(x, y_2, linestyle=':')
        plt.axvline(x=i - left, color='r')
        plt.plot(np.arange(0, right - left), shock_value[left:right])

        plt.subplot(2, 1, 2)
        origin_left, origin_right = left * race_util.step, right * race_util.step
        plt.plot(np.arange(origin_right - origin_left), origin_value[origin_left:origin_right])

        plt.show()


def main():
    pool = multiprocessing.Pool(processes=4)
    pool.map(process, os.listdir('./data/shock/'))

    total = 0
    for unit in os.listdir('./data/jump/'):
        range_list = np.load('./data/jump/' + unit)

        total += len(range_list)

    print('total = ', total) # 399172


if __name__ == '__main__':
    race_util.config()

    main()
    # view('XX.YZP.2008213000000.npy')
