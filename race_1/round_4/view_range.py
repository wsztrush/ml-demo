import numpy as np
import os
import race_util
import random
import tensorflow as tf

from build_cnn import CnnModel
from obspy import read
from matplotlib import pyplot as plt


def process(unit):
    shock_value = np.load('./data/shock/' + unit)
    range_value = np.load('./data/range/' + unit)
    origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHN')[0].data

    for left, right in range_value:
        before_left = race_util.get_before_left(left, right)

        tmp = shock_value[before_left:right]
        shock_limit = max(np.mean(shock_value[left - 10:left]) * 2, np.max(tmp) * 0.1)
        index = np.where(tmp[left - before_left:] > shock_limit)[0]

        c = 'green'
        if len(index) > 0:
            ret = cnnModel.predict(shock_value, index[0], right)
            print(ret)
            if ret == 1:
                c = 'green'
            else:
                c = 'red'

        plt.subplot(2, 1, 1)
        plt.axvline(x=left - before_left, color='red')
        plt.axhline(y=shock_limit, color='yellow')
        if len(index) > 0:
            plt.axvline(x=index[0] + left - before_left, color='green')
        plt.plot(np.arange(len(tmp)), tmp, color=c)

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(right * 10 - before_left * 10), origin_value[before_left * 10:right * 10])

        plt.show()


def main():
    unit_list = os.listdir('./data/range/')
    random.shuffle(unit_list)
    for unit in unit_list:
        print(unit)
        process(unit)


if __name__ == '__main__':
    race_util.config()

    cnnModel = CnnModel()

    main()
