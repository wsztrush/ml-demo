import numpy as np
import os
import time
import multiprocessing
import matplotlib

from matplotlib import pyplot as plt
from obspy import read

STD_PATH = "./data/std/"
RANGE_PATH = "./data/range/"
DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"


def process(unit):
    start = time.time()

    # 预处理
    file_stds = np.load(STD_PATH + unit)
    file_std = np.sqrt(np.square(file_stds[0]) + np.square(file_stds[1]))
    file_std_mean = np.mean(file_std.reshape(-1, 5), axis=1)
    file_data = read(DIR_PATH + unit[:-4] + ".BHZ")

    # 找到连续的震动较大的点
    index = np.where(file_std_mean > 50)[0]

    # 找到相邻的振幅都较大的位置
    tmp = index[1:] - index[:-1] - 3
    tmp = np.where(tmp > 0)[0]

    # 根据直观的经验做一些过滤，然后生成对应的范围文件
    last_index = 0
    result = []
    for i in tmp:
        if i - last_index > 30:
            left, right = index[last_index + 1] * 5, index[i - 1] * 5
            d = int((right - left) * 0.3)
            left = max(left - d, 0)

            std_max = np.max(file_std[left:right])
            if std_max > 700:
                result.append((left, right))

                # plt.axhline(y=int(np.max(file_std[left:right]) * 0.5), color='orange')
                # plt.plot(np.arange(right - left), file_std[left:right])
                # plt.show()

                if 10000 < right - left < 100000:
                    print(unit, left, right)

        last_index = i

    if len(result) > 0:
        np.save(RANGE_PATH + unit, result)

    print(unit, len(result), time.time() - start)


def main():
    unit_set = set()
    for unit in os.listdir(STD_PATH):
        unit_set.add(unit)
        # process(unit)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)
    # process("XX.HSH.2008198000000.npy")


if __name__ == '__main__':
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 8)

    main()
