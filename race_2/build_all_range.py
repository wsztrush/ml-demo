import numpy as np
import os
import time
import multiprocessing
import race_util

from obspy import read
from matplotlib import pyplot as plt


def get_before_ratio(shock_value, left, right):
    before_left = max(int(left - (right - left) * 0.1), 0)

    if left == before_left:
        return 0

    return np.mean(shock_value[before_left:left]) / (np.max(shock_value[left:right]) + 1.0)


# 处理逻辑
# -------
# 1. 拿到一个跳跃点，按照一秒钟的区间划分并求均值
# 2. 设置提升速度，每10秒提升10%，如果够不到就终止
# 3. 保存找到范围的结果（依然是shock_value的偏移量），并根据一些规则进行过滤
# -------
# a、长度
# b、最大的震动幅度
# c、之前是否足够平静
def process(unit):
    start_time = time.time()

    jump_index = np.load('./data/jump/' + unit)
    shock_value = np.load('./data/shock/' + unit)
    shock_mean_value = np.mean(shock_value.reshape(-1, 10), axis=1)

    result = []
    for j_index in jump_index:
        mean_index = int(j_index / 10)
        mean_limit_value = shock_mean_value[mean_index] + 1.0
        mean_limit_value_list = [mean_limit_value]

        for i in np.arange(1, len(shock_mean_value) - mean_index):
            if i % 10 == 0:
                mean_limit_value *= 1.1
            mean_limit_value_list.append(mean_limit_value)

            if shock_mean_value[mean_index + i] <= mean_limit_value:
                stop_index = j_index + i * 10

                if i > 5 and np.max(shock_value[j_index:stop_index]) > 800 and get_before_ratio(shock_value, j_index, stop_index) < 0.3:
                    result.append((j_index, stop_index))
                break

    if len(result) > 0:
        np.save('./data/all_range/' + unit, result)

    print('[COST]', unit, time.time() - start_time)


def main():
    # 生成所有可能的区间
    pool = multiprocessing.Pool(processes=4)
    pool.map(process, os.listdir('./data/jump/'))

    # 统计区间个数
    total = 0
    for unit in os.listdir('./data/all_range/'):
        range_list = np.load('./data/all_range/' + unit)

        total += len(range_list)

    print('[TOTAL RANGE]', total)  # 246574


if __name__ == '__main__':
    race_util.config()

    main()
