import numpy as np
import race_util
import os
import random

from matplotlib import pyplot as plt
from obspy import read


def get_ret_range(shock_value, range_value):
    ret = []

    for left, right in range_value:
        before_left = int(left - (right - left) * 0.1)
        if before_left < 0:
            continue

        tmp = shock_value[before_left:right]

        shock_limit = max(np.mean(shock_value[left - 10:left]) * 2, np.max(tmp) * 0.1)
        index = np.where(tmp[left - before_left:] > shock_limit)[0]

        if len(index) > 0:
            ret.append((index[0] + left, right))

    return ret


def process(unit):
    shock_value = np.load('./data/shock/' + unit)
    range_value = get_ret_range(shock_value, np.load('./data/range/' + unit))
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
            plt.plot(np.arange(right - new_left), shock_value[new_left:right])

            plt.subplot(plot_count, 2, 2 + j * 2)
            plt.axvline(x=int((left - new_left) * race_util.shock_step), color='r')
            plt.plot(np.arange(right * race_util.shock_step - new_left * race_util.shock_step), file_data[new_left * race_util.shock_step:right * race_util.shock_step])

        plt.show()


def save_sample(sample_file):
    # 给数据打标
    flag = dict()

    s = flag['GS.WXT.2008202000000.npy'] = []
    s += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 2, 0, 0]
    s += [0, 0, 0, 2, 0] + [0, 1, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 1, 0, 0, 1] + [0, 0, 0, 0, 0]
    s += [0, 0, 1, 0, 0] + [0, 1, 1, 1, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s += [1, 1, 1, 1, 1] + [0, 0, 0, 1, 1] + [0, 1, 1, 0, 1] + [1, 1, 0, 1, 1]
    s += [0, 1, 1, 2, 1] + [0, 1, 0, 2, 1] + [0, 1, 1, 1, 1] + [2, 1, 1, 1, 0]
    s += [0, 0, 0, 0, 1] + [0, 1, 1, 0, 0] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 0]
    s += [0, 0, 1, 0, 0] + [0, 0, 0, 1, 2] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s += [0, 0, 0, 0, 2] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 0, 2, 0, 0]
    s += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 2] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s += [0, 0, 0, 0, 0] + [0, 2, 1, 1, 1] + [1, 2, 1, 0, 2] + [2, 1, 2, 1, 0]
    s += [1, 1, 1, 1, 2] + [1, 0, 0, 1, 0] + [0, 1, 0, 0, 2] + [1, 1, 1, 1, 2]
    s += [0, 0, 0, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 0] + [2, 1, 1, 1, 1]
    s += [0, 0, 1, 2, 0] + [0, 0, 1, 1, 2] + [1, 1, 0, 1, 2] + [1, 1, 0, 2, 0]
    s += [1, 1, 1, 1, 0] + [1, 2, 1, 0, 0] + [1, 2, 1, 2, 1] + [1, 1, 2, 2, 2]
    s += [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2]
    s += [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2]
    s += [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2] + [2, 2] + []

    s = flag['XX.SPA.2008190000000.npy'] = []
    s += [2, 1, 0, 0, 1] + [2, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s += [0, 0, 2, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 1, 0, 0]
    s += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 1, 1, 0] + [0, 0, 0, 0, 0]
    s += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [1, 1, 0, 0, 0] + [0, 0, 1, 0, 1]
    s += [0, 0, 0, 0, 1] + [0, 0, 0, 0, 1] + [1, 1, 0, 1, 1] + [0, 0, 0, 0, 1]
    s += [2, 2, 1, 1, 1] + [2, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 0]

    s = flag['XX.HSH.2008189000000.npy'] = []
    s += [0, 2, 0, 0, 0] + [0, 2, 0, 0, 0] + [2, 2, 2, 1, 1] + [0, 0, 0, 1, 0]
    s += [0, 0, 0, 0, 0] + [0, 0, 0, 1, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s += [1, 1, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [1, 0, 0, 0, 0]
    s += [1, 0, 0, 0, 0] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1]
    s += [0, 0, 0, 1, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 1, 0] + [1, 1, 1, 0, 0]
    s += [0, 1, 2, 0, 2] + [0, 0, 0, 0, 0] + [0, 0, 0, 1, 0] + [0, 0, 0, 0, 0]
    s += [0, 0, 0, 0, 0] + [0, 0, 0, 2, 0] + [0, 0, 0, 2, 0] + [0, 0, 1, 1, 0]
    s += [0, 0, 0, 0, 1] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 2] + [0, 1, 0, 0, 0]
    s += [0, 0, 2, 0, 0] + [0, 1, 0, 1, 0] + [0, 1, 0, 0, 0] + [0, 1, 0, 0, 0]
    s += [0, 0, 0, 1, 0] + [0, 0, 1, 1, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s += [1, 1, 1, 1] + [] + [] + []

    s = flag['XX.JMG.2008188000000.npy'] = []
    s += [1, 1, 1, 1, 1] + [0, 1, 2, 1, 2] + [0, 1, 1, 1, 1] + [0, 1, 0, 2, 1]
    s += [1, 0, 2, 1, 1] + [1, 0, 0, 1, 1] + [0, 0, 0, 1, 1] + [1, 1, 0, 1, 0]
    s += [2, 1, 1, 0, 0] + [2, 1, 0, 2, 1] + [0, 1, 0, 0, 0] + [0, 0, 1, 1, 0]
    s += [1, 0, 1, 0, 1] + [2, 0, 1, 0, 0] + [1, 1, 1, 1, 1] + [0, 2, 1, 1, 1]
    s += [0, 1, 1, 1, 1] + [1, 1, 2, 0, 0] + [0, 1, 1, 0, 1] + [1, 2, 1, 0, 0]
    s += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [2, 1, 1, 1, 1] + [2, 2, 1, 1, 1]
    s += [1, 1, 0, 1, 1] + [1, 1, 2, 0, 0] + [2, 2, 0, 0, 1] + [0, 1, 1, 2, 1]
    s += [1, 0, 1, 1, 1] + [1, 0, 1, 0, 2] + [1, 1, 1, 1, 2] + [2, 1, 1, 0, 1]
    s += [1, 1, 1, 1, 2] + [0, 1, 1, 0, 2] + [2, 1, 0, 1, 2] + [1, 2, 1, 1, 0]
    s += [1, 1, 1, 1, 1] + [1, 1, 1, 2, 1] + [1, 1, 0, 1, 1] + [1, 1, 1, 1, 1]
    s += [1, 1, 1, 1, 1] + [1, 1, 2, 2, 1] + [0, 0, 1, 1, 1] + [1, 1, 1, 1, 1]
    s += [2, 1, 1, 1, 1] + [1, 0, 2, 2, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 0, 0]
    s += [0, 0, 0, 0, 0] + [1, 1, 0, 1, 2] + [] + []

    s = flag['XX.JMG.2008206000000.npy'] = []
    s += [1, 0, 0, 0, 1] + [1, 1, 1, 0, 0] + [1, 0, 2, 2, 2] + [2, 2, 1, 2, 0]
    s += [2, 2, 1, 1, 0] + [1, 0, 2, 2, 1] + [2, 2, 0, 2, 2] + [0, 0, 0, 0, 0]
    s += [0, 0, 0, 1, 2] + [1, 1, 1, 2, 0] + [1, 1, 1, 1, 2] + [0, 0, 0, 1, 1]
    s += [1, 1, 0, 1, 0] + [2, 2, 0, 1, 0] + [1, 1, 1, 1, 0] + [1, 0, 0, 0, 1]
    s += [1, 0, 1, 0, 0] + [0, 0, 0, 1, 0] + [0, 1, 1, 1, 1] + [1, 1, 1, 0, 1]
    s += [2, 2, 2, 1, 0] + [0, 0, 0, 1, 1] + [1, 0, 0, 0, 0] + [0, 0, 0, 1, 0]
    s += [2, 2, 2, 0, 2] + [0, 1, 1, 0, 1] + [1, 0, 0, 0, 1] + [0, 0, 1, 1, 0]
    s += [1, 1, 1, 2, 1] + [0, 0, 0, 1, 1] + [1, 1, 1, 0, 0] + [0, 0, 1, 0, 0]
    s += [0, 1, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 1, 1, 0, 0] + [1, 1, 1, 0, 1]
    s += [1, 1, 2, 0, 1] + [1, 1, 2, 1, 1] + [0, 1, 0, 1, 1] + [1, 0, 1, 1, 2]
    s += [0, 1, 1, 0, 1] + [0, 1, 1, 1, 0] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s += [1, 0, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [0, 1, 2, 1, 2]
    s += [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 2] + [1, 1, 1, 1, 1]
    s += [1, 1, 1, 1, 1] + [0, 1, 1, 1, 1] + [1, 0, 1, 0, 0] + [0, 0, 0, 0, 1]
    s += [0, 0, 0, 1, 1] + [2, 0, 0, 0, 1] + [0] + []

    s = flag['XX.JMG.2008186000000.npy'] = []
    s += [0, 0, 1, 2, 1] + [1, 1, 0, 1, 1] + [1, 1, 1, 1, 2] + [1, 1, 1, 1, 1]
    s += [1, 0, 1, 1, 1] + [2, 1, 0, 1, 0] + [0, 0, 1, 1, 1] + [0, 1, 1, 1, 1]
    s += [0, 0, 1, 2, 1] + [1, 2, 1, 1, 0] + [0, 0, 2, 1, 1] + [1, 0, 0, 1, 1]
    s += [0, 0, 0, 2, 1] + [0, 1, 0, 2, 2] + [0, 2, 0, 0, 1] + [0, 0, 1, 1, 1]
    s += [1, 1, 0, 2, 0] + [0, 1, 1, 1, 2] + [1, 1, 1, 0, 1] + [1, 1, 0, 0, 1]
    s += [1, 1, 0, 0, 1] + [1, 0, 0, 0, 1] + [0, 0, 0, 0, 0] + [1, 1, 0, 0, 0]
    s += [1, 0, 1, 0, 0] + [0, 1, 0, 0, 0] + [1, 0, 0, 1, 1] + [1, 1, 0, 0, 0]
    s += [1, 1, 0, 0, 1] + [1, 1, 1, 1, 1] + [0, 0, 1, 1, 1] + [0, 0, 1, 0, 1]
    s += [1, 1, 0, 1, 1] + [1, 1, 1, 0, 1] + [1, 1, 1, 0, 0] + [1, 0, 0, 1, 0]
    s += [1, 1, 1, 1, 1] + [0, 1, 0, 0, 1] + [1, 1, 1, 1, 1] + [1, 1, 1, 1, 1]
    s += [1, 1, 1, 1, 2] + [1, 1, 1, 1, 0] + [1, 1, 1, 1, 1] + [0, 1, 2, 2, 0]
    s += [1, 1, 1, 0, 1] + [0, 1, 0, 1, 0] + [1, 0, 2, 0, 0] + [1, 0, 1, 1, 1]
    s += [1, 1, 1, 1, 1] + [0, 1, 0, 1, 1] + [1, 0, 1, 0, 0] + [0, 0, 0, 1, 0]
    s += [1, 1, 2, 1, 0] + [0, 0, 1, 0, 0] + [0, 0] + []
    s += [] + [] + [] + []
    s += [] + [] + [] + []
    s += [] + [] + [] + []
    s += [] + [] + [] + []
    s += [] + [] + [] + []
    s += [] + [] + [] + []
    s += [] + [] + [] + []
    s += [] + [] + [] + []

    # 生成并保存范围文件
    ret = []
    if os.path.exists(sample_file):
        ret = np.load(sample_file)

    for u, f in flag.items():
        s = [u]

        shock_value = np.load('./data/shock/' + u)
        range_value = get_ret_range(shock_value, np.load('./data/range/' + u))

        for i in np.arange(len(f)):
            if f[i] in [0, 1]:
                left, right = range_value[i]
                s.append((left, right, f[i]))

        if len(s) > 0:
            ret.append(s)

    np.save(sample_file, ret)


if __name__ == '__main__':
    race_util.config()

    # process('XX.JMG.2008186000000.npy')

    save_sample('./data/range_sample.npy')
    # total = 0
    # unit_list = os.listdir(race_util.range_path)
    # random.shuffle(unit_list)
    # for unit in unit_list:
    #     range_value = np.load(race_util.range_path + unit)
    #     total += len(range_value)
    #     print(unit, len(range_value))
    #
    # print(total)
