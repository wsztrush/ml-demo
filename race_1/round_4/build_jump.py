import numpy as np
import os
import multiprocessing
import race_util

from matplotlib import pyplot as plt
from obspy import read

shock_limit = 1 / 1.5


def find_jump_point(shock_value, step):
    shock_value = shock_value[step:step - 10]

    tmp = np.mean(shock_value.reshape(-1, 10), axis=1) + 1
    index_1 = np.where(tmp[:-1] / tmp[1:] < shock_limit)[0]
    index_2 = np.where(tmp[:-2] / tmp[2:] < shock_limit)[0]
    index_3 = np.where(tmp[:-3] / tmp[3:] < shock_limit)[0]

    return set(index_1) & set(index_2) & set(index_3)


def process(unit):
    shock_value = np.load('./data/shock/' + unit)

    tmp = set()
    for step in np.arange(10):
        tmp |= find_jump_point(shock_value, step)

    ret = []
    for i in tmp:
        i *= 10
        if np.max(shock_value[i:i + 300]) < 800:
            continue
        ret.append(i)

    print(unit, len(ret))

    if len(ret) > 0:
        np.save('./data/jump/' + unit, sorted(ret))


def main():
    unit_list = os.listdir('./data/shock/')

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_list)


def view():
    # total = 0
    # unit_list = os.listdir('./data/jump/')
    # for unit in unit_list:
    #     jump_point_list = np.load('./data/jump/' + unit)
    #     print(unit, len(jump_point_list))
    #     total += len(jump_point_list)
    #
    # print(total)


    unit = 'XX.JJS.2008186000000.npy'
    index = np.load('./data/jump/' + unit)
    shock_value = np.load('./data/shock/' + unit)
    stock_mean_value = np.mean(shock_value.reshape(-1, 10), axis=1)
    origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

    for i in index:
        left, right, mean_index = i - 100, i + 500, int(i / 10)

        if left < 0:
            continue

        plt.subplot(2, 1, 1)

        print('[START]', stock_mean_value[int(i / 10)])

        v = stock_mean_value[int(i / 10)]
        limit_x = np.arange(100, 600, 10)
        limit_y = np.zeros(50)
        right_point = 0
        for j in np.arange(50):
            v = v * 1.01
            limit_y[j] = v

            if j != 0 and right_point == 0 and stock_mean_value[mean_index + j] < v:
                right_point = j * 10 + 100

        plt.plot(limit_x, limit_y, color='yellow')
        plt.axvline(x=i - left, color='r')
        plt.axvline(x=right_point, color='g')
        plt.plot(np.arange(0, right - left), shock_value[left:right])

        tmp = np.mean(shock_value[left:right].reshape(-1, 10), axis=1)
        plt.plot(np.arange(0, right - left, 10), tmp)

        plt.subplot(2, 1, 2)
        origin_left, origin_right = left * race_util.shock_step, right * race_util.shock_step
        plt.plot(np.arange(origin_right - origin_left), origin_value[origin_left:origin_right])

        plt.show()


if __name__ == '__main__':
    race_util.config()

    # main()
    view()
