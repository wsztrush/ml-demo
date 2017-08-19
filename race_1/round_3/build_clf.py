import numpy as np
import os
import time
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier

STD_PATH = "./data/std/"
RANGE_PATH = "./data/range/"
MODEL_FILE = "./data/clf"


def test():
    unit = "SN.LUYA.2008204000000.npy"

    clf = joblib.load(MODEL_FILE)

    file_std = np.load(STD_PATH + unit)[2]
    file_range = np.load(RANGE_PATH + unit)

    for lr in file_range:
        left, right = lr[0], lr[1]

        d_size = 20
        right += d_size - (right - left) % d_size

        tmp = file_std[left:right]
        tmp_max = np.max(tmp)
        tmp = tmp.reshape(d_size, -1)
        tmp = np.mean(tmp, axis=1)
        tmp = tmp / (tmp_max + 1.0)

        color = 'r'
        if clf.predict([tmp]) == 1:
            color = 'g'

        plt.plot(np.arange(right - left), file_std[left:right], color)
        plt.show()


def train():
    y = [
        0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 1, 1,
        0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
        1, 0,
    ]

    unit = "XX.HSH.2008198000000.npy"

    file_std = np.load(STD_PATH + unit)[2]
    file_range = np.load(RANGE_PATH + unit)

    x = []
    for lr in file_range:
        left, right = lr[0], lr[1]

        # plt.plot(np.arange(right - left), file_std[left:right])
        # plt.show()

        d_size = 20
        right += d_size - (right - left) % d_size

        tmp = file_std[left:right]
        tmp_max = np.max(tmp)
        tmp = tmp.reshape(d_size, -1)
        tmp = np.mean(tmp, axis=1)
        tmp = tmp / (tmp_max + 1.0)

        x.append(tmp)

    clf = GradientBoostingClassifier()
    clf.fit(x, y)

    joblib.dump(clf, MODEL_FILE)

    print(clf.predict(x))

    print(clf)


if __name__ == '__main__':
    # train()
    test()
    # total = 0
    # for unit in os.listdir(RANGE_PATH):
    #     file_range = np.load(RANGE_PATH + unit)
    #     total += len(file_range)
    # print(total)
