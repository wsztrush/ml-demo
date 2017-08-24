import numpy as np
import race_util
import matplotlib

from matplotlib import pyplot as plt
from obspy import read


def process(unit):
    file_std = np.load(race_util.STD_PATH + unit)
    file_std = np.sqrt(np.square(file_std[0]) + np.square(file_std[1]))
    file_range = np.load(race_util.PRE_RANGE_PATH + unit)

    file_data = read(race_util.DIR_PATH + unit[:-4] + ".BHE")[0].data

    print('total', len(file_range))

    plot_count = 5
    for i in np.arange(0, len(file_range), plot_count):
        if i % 20 == 0:
            print('-----')
        print(int(i / 5))

        for j in np.arange(min(len(file_range) - i, plot_count)):
            lr = file_range[i + j]
            left, right = lr[0], lr[1]
            new_left = max(int(left - (right - left) * 0.2), 0)

            plt.subplot(plot_count, 2, 1 + j * 2)
            plt.axvline(x=left - new_left, color='r')
            plt.plot(np.arange(right - new_left), file_std[new_left:right])

            plt.subplot(plot_count, 2, 2 + j * 2)
            plt.axvline(x=int((left - new_left) * 5), color='r')
            plt.plot(np.arange(right * 5 - new_left * 5), file_data[new_left * 5:right * 5])

        plt.show()


def save_sample():
    ret = []

    s1 = ['SN.MIAX.2008214000000.npy']
    s1 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 0]
    s1 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 1, 1]
    s1 += [0, 0, 0, 0, 0] + [0, 1, 0, 1, 0] + [0, 1, 0, 0, 0] + [1, 0, 0, 0, 0]
    s1 += [0, 0, 0, 1, 0] + [0, 0, 1, 0, 0] + [0, 1, 0, 0, 0] + [0, 0, 0, 0, 0]
    s1 += [0, 0, 0, 0, 0] + [] + [] + []
    ret.append(s1)

    s2 = ['XX.XJI.2008206000000.npy']
    s2 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 1, 0, 0, 0] + [0, 0, 0, 0, 0]
    s2 += [0, 0, 0, 0, 0] + [1, 0, 0, 0, 0] + [0, 1, 0, 1, 0] + [0, 0, 0, 1, 1]
    s2 += [0, 0, 0, 0, 0] + [1, 0, 1, 0, 1] + [0, 0, 1, 0, 0] + [0, 0, 1, 0, 0]
    s2 += [0, 0, 0, 0, 0] + [1, 0, 0, 1, 1] + [0, 0, 1, 1, 1] + [0, 0, 0, 0, 0]
    s2 += [0, 0, 1] + [] + [] + []
    ret.append(s2)

    s3 = ['XX.XCO.2008208000000.npy']
    s3 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 1, 0]
    s3 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 2, 2] + [0, 0, 0, 0, 0]
    s3 += [2, 0, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 0]
    s3 += [0, 0, 0, 0, 0] + [0, 0, 0, 1, 0] + [1, 0] + []
    ret.append(s3)

    s4 = ['XX.JMG.2008198000000.npy']
    s4 += [1, 0, 1, 0, 1] + [0, 1, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 1, 1]
    s4 += [0, 0, 1, 1, 1] + [0, 0, 0, 1, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s4 += [0, 0, 1, 1, 1] + [0, 0, 1, 0, 1] + [0, 0, 0, 1, 0] + [0, 1, 0, 0, 0]
    s4 += [0, 0, 1, 0, 0] + [0, 0, 1, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 1, 1, 0]
    s4 += [0, 1, 1, 0, 1] + [1, 0, 1, 1, 1] + [0, 0, 1, 0, 0] + [1, 1]
    ret.append(s4)

    s5 = ['XX.HSH.2008183000000.npy']
    s5 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [1, 1, 1, 0, 1]
    s5 += [0, 1, 1, 0, 1] + [1, 0, 1, 0, 0] + [1, 1, 1, 0, 0] + [1, 0, 0, 0, 0]
    s5 += [0, 1, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s5 += [0, 0, 1, 0, 0] + [0, 1, 1, 1, 0] + [0, 0, 0, 0, 0] + [0, 0, 1, 0, 1]
    s5 += [1, 1, 1, 1, 1] + [1, 0, 0, 1, 0] + [0, 0, 0, 1, 1] + [0, 0, 1, 0, 0]
    s5 += [1, 0, 1, 0, 0] + [1, 0, 0, 0, 0] + [1, 0, 0, 1, 0] + [0, 0, 0, 1, 0]
    s5 += [1, 1, 0, 1, 0] + [] + [] + []
    ret.append(s5)

    s6 = ['XX.JMG.2008204000000.npy']
    s6 += [1, 1, 0, 0, 1] + [0, 0, 0, 1, 0] + [0, 1, 1, 0, 0] + [1, 0, 1, 0, 0]
    s6 += [1, 1, 1, 0, 1] + [1, 0, 0, 1, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1]
    s6 += [0, 0, 1, 0, 0] + [0, 0, 1, 0, 0] + [0, 0, 0, 1, 0] + [1, 1, 0, 0, 1]
    s6 += [1, 0, 0, 1, 1] + [0, 0, 0, 1, 0] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s6 += [1, 1, 1, 0] + [] + [] + []
    ret.append(s6)

    s7 = ['XX.MXI.2008189000000.npy']
    s7 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s7 += [0, 0, 1, 0, 1] + [0, 1, 0, 0, 0] + [0, 1, 0, 1, 0] + [0, 1, 0, 1, 0]
    s7 += [1, 1, 0, 0, 0] + [1, 1, 1, 0, 1] + [1, 1, 1, 0, 1] + [1, 0, 1, 0, 1]
    s7 += [0, 0, 1, 1, 1] + [1, 1, 0, 1, 1] + [1, 1, 0, 0, 0] + [1, 0, 0, 1, 1]
    s7 += [0, 0, 0, 1, 1] + [0, 0, 1, 1, 1] + [1, 0, 0, 0, 1] + [1, 1, 1, 0, 0]
    s7 += [0, 1, 1, 1, 1] + [1, 1, 1, 1, 0] + [0, 0, 0, 0, 1] + [1, 1, 1, 1, 0]
    s7 += [0, 1, 0, 0, 1] + [1, 0, 0, 1, 1] + [0, 1, 0, 1, 0] + [1, 0, 0, 0, 0]
    s7 += [1, 0, 0, 0, 0] + [0, 1, 0, 0, 0] + [1, 0, 0, 0, 0] + [0, 1, 0, 1, 0]
    s7 += [1, 0, 0, 1, 0] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 1, 1, 1, 0]
    s7 += [1, 1, 1, 1, 0] + [1, 0, 0, 0, 0] + [0, 1, 0, 0, 0] + [0, 1, 0, 1, 0]
    s7 += [0, 0, 0, 0, 0] + [0, 1, 1, 1, 0] + [1, 0, 0, 0, 1] + [0, 0, 1, 1, 0]
    s7 += [0, 1, 0, 1, 1] + [0, 0, 0, 1, 1] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 0]
    s7 += [1, 1, 1, 1, 1] + [0, 0, 1, 0, 0] + [1, 1, 0, 1, 0] + [0, 0, 1, 1, 1]
    s7 += [1, 1, 1, 0, 1] + [1, 1, 0, 0, 0] + [0, 0, 1, 0, 1] + [0, 0, 0, 0, 0]
    s7 += [0, 1, 0, 1, 1] + [1, 1, 0, 1, 0] + [0, 1, 0, 1, 1] + [0, 1, 0, 1, 0]
    s7 += [1, 1, 1, 0, 0] + [0, 0, 1, 1, 1] + [1, 1, 1, 1, 1] + [0, 1, 1, 1, 1]
    s7 += [0, 1, 0, 0, 0] + [0, 0, 0, 1, 0] + [1, 1, 1, 0, 1] + [1, 0, 1, 0, 0]
    s7 += [0, 0, 1, 1, 1] + [0, 0, 0, 0, 0] + [1, 0, 1, 0, 0] + [0, 1, 0, 0, 0]
    s7 += [1, 0, 0, 0, 0] + [1, 1, 0, 0] + [] + []
    ret.append(s7)

    s8 = ['XX.JMG.2008194000000.npy']
    s8 += [0, 0, 0, 1, 0] + [0, 1, 1, 0, 0] + [0, 0, 1, 1, 1] + [0, 1, 0, 0, 1]
    s8 += [0, 0, 0, 0, 1] + [1, 0, 0, 1, 0] + [1, 1, 0, 0, 0] + [0, 0, 1, 1, 0]
    s8 += [0, 0, 0, 1, 0] + [0, 0, 1, 0, 0] + [1, 0, 0, 0, 0] + [0, 1, 0, 0, 1]
    s8 += [1, 0, 0, 0, 1] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 1, 0, 1, 0]
    s8 += [0, 1, 0, 1, 0] + [1, 0, 1, 0, 0] + [0, 1, 0, 0, 1] + [0, 0, 0, 0, 1]
    s8 += [1, 0, 0, 0, 0] + [0, 0, 0, 1, 0] + [0, 0, 0, 1] + []
    ret.append(s8)

    s9 = ['XX.PWU.2008194000000.npy']
    s9 += [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s9 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s9 += [0, 0, 0, 0, 0] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s9 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s9 += [0, 0, 0, 0, 0] + [0, 0, 0, 1, 0] + [0, 0, 1, 0, 1] + [0, 0, 0, 0, 0]
    s9 += [0, 0, 0, 0, 0] + [0, 1, 1, 1, 0] + [0, 0, 0, 0, 0] + [1, 0, 0, 0, 0]
    s9 += [0, 0, 0, 0, 0] + [0, 1, 0, 0, 0] + [0, 0, 0, 1, 0] + [0, 0, 0, 1, 0]
    s9 += [0, 0, 0, 0, 1] + [0, 1, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 0]
    s9 += [0, 1, 1, 0, 0] + [0, 0, 0, 1, 0] + [1, 1, 1, 0, 0] + [0, 0, 0, 1, 0]
    s9 += [1, 1, 1, 0, 0] + [0, 0, 1, 0, 0] + [0, 1, 0, 0, 0] + [0, 0, 0, 0, 1]
    s9 += [0, 0, 0, 0, 0] + [1, 0, 0, 1, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1]
    s9 += [1, 0, 0, 0, 0] + [0, 1, 0, 0, 0] + [1, 1, 0, 0, 1] + [0, 0, 0, 0, 0]
    s9 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 1, 1, 1, 0] + [1, 1, 0, 1, 0]
    s9 += [0, 0, 0, 0, 0] + [0, 1, 1, 1, 0] + [1, 0, 1, 1, 0] + [0, 0, 0, 0, 0]
    s9 += [0, 0, 0, 0, 0] + [1, 1, 0, 0, 0] + [0, 1, 0, 1, 1] + [0, 0, 0, 0, 0]
    s9 += [0, 0, 0, 0, 0] + [0, 0, 1, 0, 0] + [1, 0, 0, 0, 0] + [1, 0, 0, 1, 0]
    s9 += [0, 0, 0, 1, 1] + [1, 0, 0, 1, 1] + [1, 0, 0, 0, 0] + [1, 0, 0, 0, 0]
    s9 += [1, 1, 0, 0, 0] + [1, 0, 0, 0, 0] + [1, 0, 1, 1, 1] + [1, 1, 1, 1, 0]
    s9 += [1, 0, 0, 0, 0] + [0, 0, 1, 0, 1] + [0, 0, 1, 1, 0] + [1, 0, 0, 0, 0]
    s9 += [0, 0, 0, 0, 0] + [0, 0, 0, 1, 1] + [1, 1, 1, 0, 0] + [1, 1, 0, 0, 0]
    s9 += [1, 0, 1, 1, 1] + [0, 1, 0, 1, 0] + [0, 1, 0, 0, 0] + [0, 0, 0, 1, 1]
    s9 += [1, 0, 1, 1, 1] + [0, 0, 0, 1, 0] + [0, 1, 1, 0, 0] + [1, 1, 1, 0, 0]
    s9 += [0, 0, 0, 0, 1] + [0, 0, 0, 0, 1] + [0, 1, 1, 1, 1] + [0, 0, 0, 1, 0]
    s9 += [1, 1, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 1, 0, 1, 1] + [1, 1, 0, 1, 0]
    s9 += [1, 0, 1, 0, 1] + [0, 0, 0, 0, 0] + [1, 1, 1, 0, 1] + [0, 0, 1, 1, 1]
    ret.append(s9)

    s10 = ['GS.WXT.2008192000000.npy']
    s10 += [1, 0, 0, 1, 0] + [1, 0, 0, 0, 1] + [0, 0, 0, 0, 1] + [0, 1, 0, 0, 0]
    s10 += [1, 1, 1, 0, 0] + [0, 0, 0, 0, 0] + [1, 0, 1, 1, 1] + [0, 0, 1, 0, 0]
    s10 += [0, 0, 1, 1, 1] + [0, 0, 0, 0, 0] + [0, 1, 1, 0, 1] + [0, 0, 0, 0, 0]
    s10 += [0, 0, 0, 1, 0] + [1, 0, 0, 0, 0] + [1, 0, 0, 0, 0] + [0, 1, 0, 0, 0]
    s10 += [0, 0, 0, 0, 0] + [0, 0, 0, 1, 0] + [0, 0, 1, 0, 0] + [0, 0, 0, 1, 0]
    s10 += [0, 0, 1, 0, 1] + [1, 0, 0, 0, 1] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 1]
    s10 += [0, 0, 0, 1, 0] + [0, 0, 1, 1, 0] + [1, 2, 2, 2, 2] + [2, 2, 2, 2, 2]
    s10 += [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2] + [2, 2, 2, 1, 0] + [0, 0, 1, 1, 1]
    s10 += [0, 0, 1, 1, 0] + [0, 0, 1, 0, 1] + [0, 0, 0, 0, 0] + [1, 1, 0, 1, 0]
    s10 += [0, 1, 0, 0, 0] + [1, 0, 0, 0, 1] + [0, 0, 0, 0, 0] + [0, 0, 1, 0, 0]
    s10 += [1, 0, 1, 0, 0] + [0, 1, 0, 0, 0] + [1, 1, 1, 0, 0] + [0, 0, 1, 0, 1]
    s10 += [1, 0, 1, 1, 0] + [0, 0] + [] + []
    ret.append(s10)

    s11 = ['XX.MXI.2008184000000.npy']
    s11 += [1, 1, 1, 0, 1] + [0, 0, 0, 0, 1] + [1, 1, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 1, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 1, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [1, 0, 0, 0, 1] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 1, 0, 0] + [0, 1, 1, 0, 0] + [0, 0, 0, 0, 0] + [1, 0, 0, 0, 0]
    s11 += [1, 0, 0, 0, 0] + [1, 0, 0, 0, 0] + [1, 0, 0, 0, 1] + [0, 0, 0, 0, 1]
    s11 += [0, 0, 0, 0, 0] + [0, 1, 1, 0, 0] + [0, 1, 0, 0, 0] + [0, 1, 1, 1, 0]
    s11 += [0, 0, 0, 0, 1] + [1, 1, 0, 0, 0] + [1, 0, 0, 0, 0] + [1, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [1, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 1, 0, 0, 1] + [1, 1, 0, 0, 0] + [0, 0, 1, 0, 0]
    s11 += [0, 0, 1, 0, 0] + [0, 1, 0, 1, 0] + [1, 0, 0, 0, 0] + [0, 0, 0, 1, 1]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 1, 1] + [1, 1, 1, 1, 0] + [1, 1, 0, 0, 1] + [1, 1, 1, 0, 1]
    s11 += [0, 0, 0, 0, 0] + [0, 1, 0, 0, 1] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 3, 0, 1] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 0, 0, 0, 1]
    s11 += [0, 0, 1, 0, 1] + [1, 1, 0, 0, 0] + [1, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 1, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 1, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [2, 2, 2, 2, 2]
    s11 += [2, 2, 2, 2, 2] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 1, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 1, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [1, 0, 0, 0, 0] + [1, 0, 0, 0, 0] + [0, 1, 0, 0, 0]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0] + [0, 0, 0, 0, 1] + [0, 0, 1, 1, 0]
    s11 += [0, 0, 1, 0, 1] + [0, 0, 1, 1, 0] + [1, 0, 0, 0, 1] + [0, 1, 1, 0, 0]
    s11 += [0, 1, 0, 1, 0] + [0, 0, 0, 1, 1] + [0, 1, 1, 1, 1] + [1, 0, 1, 1, 1]
    s11 += [1, 1, 1, 0, 0] + [1, 1, 1, 1, 1] + [1, 0, 1, 0, 1] + [0, 1, 0, 0, 0]
    s11 += [1, 0, 1, 1, 1] + [1, 1, 1, 1, 1] + [1, 1, 0, 0, 1] + [1, 0, 0, 1, 0]
    s11 += [0, 0, 0, 1, 1] + [0, 1, 0, 1, 1] + [1, 0, 0, 0, 0] + [0, 0, 1, 0, 1]
    s11 += [1, 0, 1, 1, 1] + [0, 1, 0, 1, 0] + [1, 0, 0, 1, 0] + [1, 0, 0, 1, 0]
    s11 += [0, 0, 0, 1, 1] + [0, 1, 1, 1, 1] + [1, 1, 0, 0, 1] + [0, 0, 1, 0, 1]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 1, 1, 0] + [0, 0, 0, 1, 0] + [0, 1, 1, 1, 1]
    s11 += [0, 0, 0, 0, 0] + [0, 0, 1, 0] + [] + []
    ret.append(s11)

    s12 = ['XX.YZP.2008196000000.npy']  # TODO 这个文件有问题
    s12 += [0, 1, 1, 1, 0] + [1, 1, 0, 0, 1] + [0, 0, 1, 1, 1] + [1, 0, 1, 1, 1]
    s12 += [0, 1, 1, 1, 1] + [1, 1, 1, 1, 1] + [0, 0, 0, 0, 1] + [1, 0, 0, 1, 1]
    s12 += [1, 1, 1, 1, 0] + [0, 1, 1, 1, 0] + [0, 1, 1, 0, 1] + [0, 0, 1, 1, 0]
    s12 += [1, 0, 1, 0, 0] + [1, 0, 1, 1, 1] + [0, 1, 1, 0, 0] + [1, 1, 1, 0, 1]
    s12 += [1, 1, 1, 1, 1] + [1, 1, 1, 0, 0] + [1, 1, 0, 1, 0] + [1, 1, 1, 1, 1]
    s12 += [0, 0, 0, 1, 0] + [1, 1, 1, 1, 1] + [1, 0, 0, 0, 0] + [1, 1]
    ret.append(s12)

    total = 0
    for i in ret:
        for j in i:
            if j == 0:
                total += 1
    print(total)

    np.save(race_util.MODEL_SAMPLE_FILE, ret)


if __name__ == '__main__':
    race_util.config()
    # process('XX.YZP.2008196000000.npy')
    save_sample()
