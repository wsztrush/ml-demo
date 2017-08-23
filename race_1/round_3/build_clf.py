import numpy as np
import os
import time
import random

from obspy import read
from matplotlib import animation
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier

import race_config


def check():
    clf = joblib.load(race_config.MODEL_FILE)

    # 构建图形
    fig = plt.figure()

    axs = [fig.add_subplot(211 + i) for i in np.arange(2)]
    for ax in axs:
        ax.set_ylim(0, 10000)
        ax.set_xlim(0, 10)

    lines = [ax.plot([], [])[0] for ax in axs]

    # 获取值
    def next_value():
        unit_list = os.listdir(race_config.PRE_RANGE_PATH)
        random.shuffle(unit_list)

        for unit in unit_list:
            print('----', unit, '----')
            file_std, file_range = get_file_std_range(unit)

            file_content = read(race_config.DIR_PATH + unit[:-4] + '.BHZ')
            file_data = file_content[0].data

            for lr in file_range:
                left, right = lr[0], lr[1]

                x = get_x(file_std, left, right)

                if clf.predict([x]) == 1:
                    if right - left > 50000:
                        print(unit, left, right)
                        continue

                    a, b = file_std[left:right], file_data[left * 5: right * 5]

                    if np.max(b) == 0:
                        print(unit, left * 5, right * 5)

                    yield a, b

    # 更新展示
    def refresh(values):
        for i in np.arange(len(values)):
            lines[i].set_data(np.arange(len(values[i])), values[i])

            axs[i].set_ylim(np.min(values[i]), np.max(values[i]))
            axs[i].set_xlim(0, len(values[i]))

        return lines

    # 设置动画
    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=1000, repeat=False)
    plt.show()


def get_x(file_std, left, right):
    right += 11 - (right - left) % 11

    tmp = file_std[left:right]
    tmp_max = np.max(tmp)
    tmp = tmp.reshape(11, -1)
    tmp = np.mean(tmp, axis=1)

    ret = tmp / (tmp_max + 1.0)
    ret = ret.tolist()

    ret.append(right - left)
    ret.append(tmp_max)

    return ret


def train():
    sample_list = np.load(race_config.MODEL_SAMPLE_FILE)

    x_list = []
    y_list = []
    for sample in sample_list:
        unit = sample[0]

        file_std, file_range = get_file_std_range(unit)

        for i in np.arange(1, len(sample)):
            if sample[i] == 2:
                continue

            lr = file_range[i - 1]

            x_list.append(get_x(file_std, lr[0], lr[1]))
            y_list.append(sample[i])

    clf = GradientBoostingClassifier()
    clf.fit(x_list, y_list)

    print(clf)

    joblib.dump(clf, race_config.MODEL_FILE)


def get_file_std_range(unit):
    file_stds = np.load(race_config.STD_PATH + unit)
    file_std = np.sqrt(np.square(file_stds[0]) + np.square(file_stds[1]))

    file_range = np.load(race_config.PRE_RANGE_PATH + unit)

    return file_std, file_range


def f_count():
    clf = joblib.load(race_config.MODEL_FILE)
    total = 0

    std_max_list = []
    for unit in os.listdir(race_config.PRE_RANGE_PATH):
        print(unit)

        file_std, file_range = get_file_std_range(unit)

        for lr in file_range:
            left, right = lr[0], lr[1]

            x = get_x(file_std, left, right)

            if clf.predict([x]) == 1:
                total += 1
                std_max_list.append(np.max(file_std[left:right]))

    tmp = np.argpartition(std_max_list, -20000)[-20000:]
    print(np.array(std_max_list)[tmp])

    print(total)


if __name__ == '__main__':
    # train()
    # check()
    f_count()
