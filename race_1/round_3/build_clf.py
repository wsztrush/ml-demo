import numpy as np
import os
import time
from matplotlib import animation
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier

STD_PATH = "./data/std/"
RANGE_PATH = "./data/range/"
MODEL_FILE = "./data/clf"
MODEL_SAMPLE_FILE = "./data/clf_sample.npy"
D_SIZE = 20


def test():
    clf = joblib.load(MODEL_FILE)

    # 构建图形
    fig = plt.figure()

    axs = [fig.add_subplot(111 + i) for i in np.arange(1)]
    for ax in axs:
        ax.set_ylim(0, 10000)
        ax.set_xlim(0, 10)

    lines = [ax.plot([], [])[0] for ax in axs]

    # 获取值
    def next_value():
        for unit in os.listdir(RANGE_PATH):
            file_std = np.load(STD_PATH + unit)[2]
            file_range = np.load(RANGE_PATH + unit)

            for lr in file_range:
                left, right = lr[0], lr[1]

                x = get_x(file_std, left, right)

                if clf.predict([x]) == 1:
                    yield [file_std[left:right]]

    # 更新展示
    def refresh(values):
        for i in np.arange(1):
            lines[i].set_data(np.arange(len(values[i])), values[i])

            axs[i].set_ylim(np.min(values[i]), np.max(values[i]))
            axs[i].set_xlim(0, len(values[i]))

        return lines

    # 设置动画
    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=100, repeat=False)
    plt.show()


def get_x(file_std, left, right):
    right += D_SIZE - (right - left) % D_SIZE

    tmp = file_std[left:right]
    tmp_max = np.max(tmp)
    tmp = tmp.reshape(D_SIZE, -1)
    tmp = np.mean(tmp, axis=1)

    ret = tmp / (tmp_max + 1.0)
    ret = ret.tolist()

    ret.append(right - left)
    ret.append(tmp_max)

    return ret


def train():
    sample_list = np.load(MODEL_SAMPLE_FILE)

    x_list = []
    y_list = []
    for sample in sample_list:
        unit = sample[0]

        file_std = np.load(STD_PATH + unit)[2]
        file_range = np.load(RANGE_PATH + unit)

        for i in np.arange(1, len(sample)):
            if sample[i] == 2:
                continue

            lr = file_range[i - 1]

            x_list.append(get_x(file_std, lr[0], lr[1]))
            y_list.append(sample[i])

    clf = GradientBoostingClassifier()
    clf.fit(x_list, y_list)

    joblib.dump(clf, MODEL_FILE)


if __name__ == '__main__':
    # train()
    test()
