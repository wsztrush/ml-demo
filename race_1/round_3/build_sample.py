import numpy as np

from matplotlib import pyplot as plt
import matplotlib

D_SIZE = 20
STD_PATH = "./data/std/"
RANGE_PATH = "./data/range/"
MODEL_SAMPLE_FILE = "./data/clf_sample.npy"


def process(unit):
    file_std = np.load(STD_PATH + unit)[2]
    file_range = np.load(RANGE_PATH + unit)

    for i in np.arange(0, len(file_range), 9):
        for j in np.arange(min(len(file_range) - i, 9)):
            lr = file_range[i + j]
            left, right = lr[0], lr[1]

            plt.subplot(331 + j)
            plt.plot(np.arange(right - left), file_std[left:right])

        plt.show()


def save_sample():
    ret = []
    s1 = ['GS.WDT.2008183000000.npy']
    s1 += [1, 1, 0, 0, 1, 1, 1, 1, 1, ]
    s1 += [1, 1, 1, 0, 1, 1, 0, 1, 1, ]
    s1 += [1, 1, 1, 1, 0, 1, 1, 1, 1, ]
    s1 += [1, 1, 1, 1, 0, 1, 1]
    ret.append(s1)

    s2 = ['SN.LUYA.2008206000000.npy']
    s2 += [1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s2 += [1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s2 += [1, 1, 1, 1, 1, 1, 1, 1, 0, ]
    s2 += [1, 1, 1, 1, 1, ]
    ret.append(s2)

    s3 = ['XX.JJS.2008189000000.npy']
    s3 += [0, 0, 1, 0, 1, 1, 1, 1, 1, ]
    s3 += [0, 1, 0, 0, 1, 1, 0, 0, 1, ]
    s3 += [0, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s3 += [1, 1, 1, 1, 1, 1, 1, 0, 0, ]
    s3 += [1, 0, 0, 0, 0, 1]
    ret.append(s3)

    s4 = ['XX.MXI.2008204000000.npy']
    s4 += [1, 0, 1, 0, 0, 0, 0, 0, 1, ]
    s4 += [1, 1, 2, 1, 1, 0, 1, 0, 0, ]
    s4 += [1, 1, 1, 0, 1, 1, 1, 1, 1, ]
    s4 += [1, 1, 1, 1, 1, 1, 0, 0, 1, ]
    s4 += [0, 1, 0, 1, 1, 1, 1, 1, 0, ]
    s4 += [1, 0, 1, 0, 0, 1, 0, 1, 1, ]
    s4 += [0, 0, 0, 1, 0, 1, 0, 1, 1, ]
    s4 += [1, 0, 1, 1, 1, 1, 1, 1, 1, ]
    s4 += [1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s4 += [1, 1, 1, 1, 1, 1, 1, 0, 0, ]
    s4 += [1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s4 += [0, 1, 1, 0, 1, 0, 1, 1, 1, ]
    s4 += [0, 1, 1, 1, 0, 1, 0, 1, 1, ]
    s4 += [1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s4 += [1, 1, 0, 1, 1, 1, 1, 0, 0, ]
    s4 += [1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s4 += [1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s4 += [1, 1, 1, 1, 1, 1, 1, 0, 1, ]
    s4 += [0, 0, 1, 1, 1, 1, 1, 1, 1, ]
    s4 += [1, 1, 0, 1, 1, 1, 1, 1, 1, ]
    s4 += [1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s4 += [1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s4 += [0, 1, 1, 1, 1, 1, 1, 1, 1, ]
    s4 += [1, 1, 1, 1, 1, 1, 1, 0, 1, ]
    s4 += [1, 1, 1, 1, 1, ]
    ret.append(s4)

    np.save(MODEL_SAMPLE_FILE, ret)


def main():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 8)

    process('XX.MXI.2008204000000.npy')


if __name__ == '__main__':
    # main()
    save_sample()
