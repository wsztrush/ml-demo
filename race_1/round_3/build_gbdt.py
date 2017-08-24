import numpy as np
import os
import time
import random
import race_util

from obspy import read
from matplotlib import animation
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier


def check():
    gbdt = joblib.load(race_util.GBDT_MODEL_FILE)

    # 构建图形
    fig = plt.figure()

    axs = [fig.add_subplot(211 + i) for i in np.arange(2)]
    for ax in axs:
        ax.set_ylim(0, 10000)
        ax.set_xlim(0, 10)

    lines = [ax.plot([], [])[0] for ax in axs]

    # 获取值
    def next_value():
        unit_list = os.listdir(race_util.PRE_RANGE_PATH)
        random.shuffle(unit_list)

        for unit in unit_list:
            print('----', unit, '----')
            file_std, file_range = get_file_std_range(unit)

            file_content = read(race_util.DIR_PATH + unit[:-4] + '.BHZ')
            file_data = file_content[0].data

            for lr in file_range:
                left, right = lr[0], lr[1]

                x = get_feature(file_std, left, right)

                if gbdt.predict([x]) == 0:
                    left = max(int(left - (right - left) * 0.1), 0)

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


def get_feature(file_std, left, right):
    std_max = np.max(file_std[left:right])
    max_index = left

    # 过滤掉长尾巴
    index = np.where(file_std[left:right] > std_max * 0.1)[0]
    if len(index) > 0:
        right = left + max(index)

    # 找到最大值的位置
    index = np.where(file_std[left:right] == std_max)[0]
    if len(index) > 0:
        max_index = left + max(index)

    # 防止除数变0
    std_max += 1.0

    # 前后分别划分
    sub_before = get_sub_std(file_std[left:max_index], std_max, 3)
    sub_after = get_sub_std(file_std[max_index:right], std_max, 4)

    # 计算范围前面一点的值
    before_mean = -1
    before_location = int(left - (right - left) * 0.1)
    if before_location >= 0:
        before_mean = np.mean(file_std[before_location:left])

    # 组合特征
    ret = []
    ret += sub_before
    ret += sub_after
    ret.append((max_index - left) / (right - left))
    ret.append(np.std(sub_before) / std_max)
    ret.append(np.std(sub_after) / std_max)
    ret.append(before_mean / std_max)

    # 数据校验
    if np.isnan(ret).any() or np.isinf(ret).any():
        print(left, max_index, right)
        print(ret)

    if max_index == left:
        print('max is left', left, right)

    return ret


def get_sub_std(file_std, std_max, size):
    if len(file_std) < size * 5:
        return (np.zeros(size) - 1).tolist()

    tmp = file_std[:len(file_std) - (len(file_std) % size)]
    tmp = tmp.reshape(size, -1)
    tmp = np.mean(tmp, axis=1)

    return (tmp / std_max).tolist()


def get_file_std_range(unit):
    file_stds = np.load(race_util.STD_PATH + unit)
    file_std = np.sqrt(np.square(file_stds[0]) + np.square(file_stds[1]))

    file_range = np.load(race_util.PRE_RANGE_PATH + unit)

    return file_std, file_range


def train():
    sample_list = np.load(race_util.MODEL_SAMPLE_FILE)

    x_list = []
    y_list = []
    for sample in sample_list:
        unit = sample[0]

        file_std, file_range = get_file_std_range(unit)

        for i in np.arange(1, len(file_range)):
            if sample[i] == 2:
                continue

            lr = file_range[i - 1]

            x_list.append(get_feature(file_std, lr[0], lr[1]))
            y_list.append(sample[i])

    gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, subsample=0.6, min_samples_leaf=3)
    gbdt.fit(x_list, y_list)

    print(len(x_list))
    print(gbdt)

    joblib.dump(gbdt, race_util.GBDT_MODEL_FILE)


def stat():
    gbdt = joblib.load(race_util.GBDT_MODEL_FILE)

    total = 0
    unit_list = os.listdir(race_util.PRE_RANGE_PATH)
    random.shuffle(unit_list)

    for unit in unit_list:
        file_std, file_range = get_file_std_range(unit)
        for lr in file_range:
            left, right = lr[0], lr[1]

            x = get_feature(file_std, left, right)
            if gbdt.predict([x]) == 1:
                total += 1

    print(total)


if __name__ == '__main__':
    # train()
    # check()
    stat()