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

    s1 = ['XX.YZP.2008213000000.npy']
    s1 += [0, 0, 0, 1, 1] + [1, 1, 0, 0, 1] + [0, 1, 1, 1, 0] + [1, 1, 1, 0, 1]
    s1 += [1, 1, 0, 1, 1] + [1, 1, 1, 1, 1] + [1, 0, 1, 1, 1] + [1, 1, 1, 0, 0]
    s1 += [0, 1, 0, 0, 1] + [1, 1, 0, 1, 1] + [1, 0, 1, 1, 1] + [1, 1, 1, 1, 1]
    s1 += [0, 0, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s1 += [1, 1, 0, 0, 1] + [1, 0, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 0, 1]
    s1 += [0, 1, 1, 1, 1] + [] + [] + []
    ret.append(s1)

    s2 = ['XX.YZP.2008212000000.npy']
    s2 += [1, 0, 0, 1, 1] + [1, 1, 1, 0, 0] + [1, 1, 1, 1, 1] + [0, 1, 1, 1, 1]
    s2 += [0, 0, 0, 0, 0] + [1, 1, 1, 0, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s2 += [1, 1, 0, 1, 1] + [1, 1, 0, 1, 1] + [1, 0, 1, 1, 0] + [1, 0, 1, 1, 1]
    s2 += [1, 1, 0, 1, 1] + [0, 0, 1, 1, 1] + [0, 1, 0, 1, 1] + [1, 1, 1, 1, 1]
    s2 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [0, 0, 1, 0, 1] + [0, 1, 1, 0, 1]
    s2 += [0, 0, 1, 1, 1] + [0, 1, 0, 1, 1] + [1, 1, 1, 1, 1] + [0, 0, 1, 1, 1]
    s2 += [0, 0, 0, 1, 1] + [0, 1, 0, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s2 += [1] + [] + [] + []
    ret.append(s2)

    s3 = ['XX.XCO.2008212000000.npy']
    s3 += [0, 0, 0, 0, 0] + [1, 0, 0, 1, 0] + [1, 0, 1, 0, 1] + [1, 1, 1, 1, 0]
    s3 += [0, 1, 0, 0, 0] + [1, 0, 0, 0, 0] + [0, 0, 1, 0, 1] + [0, 0, 1, 0, 0]
    s3 += [0, 1, 0, 1, 0] + [0, 1, 0, 0, 0] + [0, 0, 1, 0, 0] + [1, 1, 1, 1, 1]
    s3 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 0] + [0]
    ret.append(s3)

    s4 = ['XX.JMG.2008197000000.npy']
    s4 += [1, 1, 1, 1, 0] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 1, 0, 0]
    s4 += [1, 0, 0, 0, 0] + [1, 1, 0, 0, 0] + [0, 1, 1, 1, 1] + [0, 1, 0, 0, 0]
    s4 += [0, 0, 0, 0, 1] + [0, 0, 0, 0, 0] + [0, 1, 1, 1, 0] + [0, 1, 0, 0, 1]
    s4 += [0, 0, 0, 0, 0] + [0, 1, 0, 1, 0] + [0, 0, 1, 0, 1] + [1, 1, 0, 0, 1]
    s4 += [0, 0, 0, 0, 0] + [1, 0, 1, 0, 1] + [1, 0, 0, 0, 1] + [0, 0, 0, 1, 0]
    s4 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 1] + [0, 1, 0, 1, 1]
    s4 += [0, 1, 0, 0, 0] + [0, 1, 0, 0, 0] + [0, 1, 0, 0, 0] + [1, 0, 0, 0, 0]
    s4 += [0, 0, 1, 1, 0] + [0, 1, 0, 1, 0] + [0, 0, 0, 0, 0] + [0, 0, 1, 0, 0]
    s4 += [0, 0, 0, 0, 1] + [0, 0, 1, 1, 0] + [1, 0, 0, 0, 1] + [0, 0, 0, 0, 0]
    s4 += [0, 1, 0, 1, 1] + [0, 1, 0, 1, 0] + [0, 1, 0, 0, 0] + [1, 0, 1, 1, 0]
    s4 += [0, 1, 1, 0, 1] + [0, 1, 1, 1, 0] + [1, 1, 0, 0, 0] + [1, 1, 0, 1, 1]
    s4 += [0, 0, 0, 1, 0] + [1, 1, 1, 1, 1] + [0, 1, 1, 1, 1] + [0, 1, 0, 0, 0]
    s4 += [0, 1, 0, 1, 1] + [0, 0, 1, 1, 0] + [1, 0, 1, 1, 1] + [0, 0, 1, 0, 1]
    s4 += [0, 0] + [] + [] + []
    ret.append(s4)

    s5 = ['XX.SPA.2008205000000.npy']
    s5 += [0, 1, 1, 1, 0] + [0, 0, 0, 1, 0] + [1, 0, 1, 0, 1] + [0, 0, 0, 0, 0]
    s5 += [1, 0, 1, 1, 1] + [0, 1, 0, 1, 1] + [0, 1, 1, 0, 1] + [1, 1, 1, 1, 1]
    s5 += [1, 0, 1, 1, 1] + [0, 0, 1, 0, 0] + [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2]
    s5 += [1, 0, 0, 0, 0] + [1, 1, 0, 0, 1] + [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2]
    s5 += [0, 0, 0, 1, 0] + [1, 0, 0, 1, 0] + [0, 0, 1, 0, 0] + [0, 1, 0, 0]
    ret.append(s5)

    total = 0
    for i in ret:
        for j in i:
            if j == 0:
                total += 1
    print(total)

    np.save(race_util.sample_1_file, ret)


def save_sample_2():
    ret = []

    s1 = ['GS.WDT.2008191000000.npy']
    s1 += [1, 2, 1, 1, 1] + [1, 1, 1, 1, 1] + [0, 0, 1, 1, 1] + [1, 1, 1, 1, 1]
    s1 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1] + []
    ret.append(s1)

    s2 = ['GS.WDT.2008184000000.npy']
    s2 += [2, 2, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s2 += [1, 0, 1, 1, 1] + [1, 1, 0, 1, 1] + [1] + []
    ret.append(s2)

    s3 = ['GS.WDT.2008195000000.npy']
    s3 += [1, 1, 1, 2, 2] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s3 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s3 += [1, 1, 1, 1, 1] + [1, 1, 1, 0, 1] + [1, 1] + []
    ret.append(s3)

    s4 = ['XX.MXI.2008188000000.npy']
    s4 += [0, 1, 0, 0, 0] + [0, 0, 0, 0, 0] + [1, 0, 1, 1, 1] + [1, 1, 0, 0, 0]
    s4 += [0, 1, 1, 1, 1] + [1, 1, 1, 1, 0] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s4 += [1, 1, 0, 1, 1] + [1, 1, 0, 1, 1] + [0, 1, 1, 1, 1] + [1, 1, 0, 1, 1]
    s4 += [2, 1, 1, 1, 1] + [1, 1, 1, 0, 2] + [1, 1, 1, 1, 1] + [0, 0, 1, 0, 1]
    s4 += [1, 1, 0, 1, 1] + [1] + [] + []
    ret.append(s4)

    s5 = ['XX.QCH.2008189000000.npy']
    s5 += [0, 0, 1, 0, 0] + [1, 1, 1, 1, 0] + [1, 0, 1, 0, 1] + [1, 1, 1, 1, 1]
    s5 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 0, 1, 1, 1] + [1, 1, 1, 1, 0]
    s5 += [1, 0, 0, 1, 1] + [1, 1, 1, 0, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s5 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s5 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 0] + [1, 1, 1, 0, 1]
    s5 += [0, 1, 1, 1, 0] + [1, 1, 1, 1, 1] + [1, 1, 1, 0, 1] + [1, 1, 1, 1, 1]
    s5 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 0]
    s5 += [1, 0, 0, 0, 1] + [1, 0, 1, 1, 0] + [1, 0] + []
    ret.append(s5)

    s6 = ['XX.YZP.2008212000000.npy']
    s6 += [1, 1, 1, 2, 1] + [1, 1, 1, 2, 1] + [1, 1, 1, 0, 1] + [1, 1, 1, 1, 1]
    s6 += [1, 2, 2, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s6 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 0, 1, 1, 1] + [1, 1, 0, 1, 1]
    s6 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 0, 1] + [1, 1, 1, 1, 1]
    s6 += [1, 0, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s6 += [0, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [0, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s6 += [1, 1, 1, 1, 1] + [1, 1, 1, 1] + [] + []
    ret.append(s6)

    s7 = ['XX.YZP.2008191000000.npy']
    s7 += [1, 1, 1, 1, 1] + [1, 1, 0, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s7 += [1, 1, 1, 0, 1] + [0, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 0, 1, 1]
    s7 += [1, 1, 1, 0, 1] + [1, 1, 1, 1, 1] + [1, 2, 1, 1, 2] + [0, 1, 1, 1, 1]
    s7 += [1, 1, 0, 1, 1] + [0, 1, 1, 1, 0] + [1, 1, 0, 0, 1] + [1, 1, 0, 1, 1]
    s7 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 0, 1, 1] + [0, 1, 1, 1, 0]
    s7 += [1, 1, 1, 1, 1] + [1, 1, 1, 0, 1] + [1, 1, 1, 0, 1] + [1, 1, 1, 0, 1]
    s7 += [1, 0, 1, 1, 1] + [1, 0, 0, 0, 1] + [1, 2, 2, 2, 2] + [2, 0, 1, 1, 1]
    ret.append(s7)

    s8 = ['XX.YZP.2008193000000.npy']
    s8 += [1, 1, 1, 0, 0] + [1, 1, 0, 1, 1] + [1, 0, 0, 1, 1] + [1, 1, 1, 1, 1]
    s8 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 2, 1] + [1, 1, 0, 1, 1]
    s8 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 2, 1] + [1, 1, 1, 1, 0]
    s8 += [0, 1, 1, 1, 0] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [0, 1, 1, 1, 1]
    s8 += [1, 1, 1, 1, 1] + [1, 1, 1, 0, 1] + [1, 1, 0, 1, 1] + [1, 1, 1, 1, 1]
    s8 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1] + []
    ret.append(s8)

    s9 = ['XX.JMG.2008185000000.npy']
    s9 += [1, 0, 2, 2, 1] + [1, 1, 1, 1, 1] + [1, 0, 1, 2, 1] + [1, 1, 1, 1, 1]
    s9 += [1, 1, 1, 1, 1] + [2, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 2, 1]
    s9 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 0, 1, 1, 1]
    s9 += [1, 0, 0, 1, 1] + [1, 0, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s9 += [1, 1, 1, 1, 1] + [1, 0, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s9 += [1, 0, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1]
    ret.append(s9)

    s10 = ['XX.YZP.2008205000000.npy']
    s10 += [1, 1, 0, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 0, 1, 0] + [1, 1, 0, 0, 1]
    s10 += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 0, 1]
    s10 += [1, 1, 1, 1, 1] + [1, 0, 1, 1, 1] + [0, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s10 += [1, 1, 1, 1, 1] + [0, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s10 += [1, 1, 0, 1, 1] + [1, 1, 0, 1, 1] + [1, 1, 1, 1, 1] + [0, 2, 2, 2, 2]
    s10 += [1, 1, 1, 1, 0] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 0, 1, 1, 1]
    s10 += [1, 1, 1, 1, 1] + [1, 0, 0, 1, 0] + [1, 1, 0, 1, 1] + [0, 1, 1, 0, 0]
    s10 += [1, 1, 1, 1, 1] + [1, 1, 1, 0, 1] + [1, 1, 1, 1, 1] + [0, 1, 1, 1, 1]
    s10 += [1, 1, 1, 0, 1] + [1, 0, 1, 1, 2] + [1, 1, 1, 1, 1] + [1, 0]
    ret.append(s10)

    total = 0
    for i in ret:
        for j in i:
            if j == 1:
                total += 1
    print(total)

    np.save(race_util.sample_2_file, ret)


if __name__ == '__main__':
    race_util.config()

    # process('XX.YZP.2008205000000.npy')

    # save_sample_1()
    # save_sample_2()

    total = 0
    unit_list = os.listdir(race_util.range_path)
    random.shuffle(unit_list)
    for unit in unit_list:
        range_value = np.load(race_util.range_path + unit)
        total += len(range_value)
        print(unit, len(range_value))

    print(total)
