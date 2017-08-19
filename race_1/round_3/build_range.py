import numpy as np
import os
import time
import multiprocessing
from matplotlib import pyplot as plt

STD_PATH = "./data/std/"
RANGE_PATH = "./data/range/"


def process(unit):
    start = time.time()

    # 预处理
    file_stds = np.load(STD_PATH + unit)
    file_std = file_stds[2]
    file_std_mean = np.mean(file_std.reshape(-1, 5), axis=1)

    # 找到连续的震动较大的点
    index = np.where(file_std_mean > 100)[0]

    # 找到相邻的振幅都较大的位置
    tmp = index[1:] - index[:-1] - 1
    tmp = np.where(tmp > 0)[0]

    # 根据直观的经验做一些过滤，然后生成对应的范围文件
    last_index = 0
    result = []
    for i in tmp:
        if i - last_index > 30:
            left, right = index[last_index + 1] * 5, index[i - 1] * 5
            left = max(left - int((right - left) * 0.2), 0)

            if np.max(file_std[left:right]) > 800:
                result.append((left, right))
                # plt.plot(np.arange(right - left), file_std[left:right])
                # plt.show()

                if right - left > 10000:
                    print(unit, left, right)

        last_index = i

    if len(result) > 0:
        np.save(RANGE_PATH + unit, result)

    print(unit, len(result), time.time() - start)


def main():
    unit_set = set()
    for unit in os.listdir(STD_PATH):
        unit_set.add(unit)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)
    # process("XX.HSH.2008198000000.npy")


if __name__ == '__main__':
    main()
