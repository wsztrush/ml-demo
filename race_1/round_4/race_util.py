import matplotlib
import numpy as np

origin_dir_path = "/Users/tianchi.gzt/Downloads/race_1/after/"
shock_path = "./data/shock/"
range_path = "./data/range/"
init_range_path = "./data/init_range/"

sample_1_file = './data/sample_1.npy'
sample_2_file = './data/sample_2.npy'

model_1 = './data/model_1'

stock_step = 10


def config():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 8)


def get_before_left(left, right):
    return max(int(left - (right - left) * 0.1), 0)


# 区间长度的有效性判断
# ******************************
# 一个地震区间的长度以及其中的最大振幅，根据猜测做一次过滤
def is_good_lr(stock_value, left, right):
    return (right - left) * stock_step > 500 and np.max(stock_value[left:right] > 1000)


# 区间过滤
# ******************************
# 连续【stock_gap】个小于【stock_limit】的位置就当做是平静的区间，利用平静的区间对原始区间进行切换
def split_range(stock_value, left, right, stock_limit=100, gap=3):
    b = stock_value[left:right]

    index = np.where(b > stock_limit)[0]
    continuity_index = np.where(index[1:] - index[:-1] - gap > 0)[0]

    last_index = 0
    result = []

    for i in continuity_index:
        l, r = index[last_index] + left, index[i] + left + 1
        if is_good_lr(stock_value, l, r):
            result.append((l, r))

        last_index = i + 1

    if last_index > 0:
        l, r = index[last_index] + left, len(b) + left
        if is_good_lr(stock_value, l, r):
            result.append((l, r))

    return result
