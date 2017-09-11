import matplotlib
import numpy as np
import build_model_1

from sklearn.externals import joblib

origin_dir_path = "/Users/tianchi.gzt/Downloads/race_1/after/"
shock_path = "./data/shock/"
range_path = "./data/range/"

model_1 = joblib.load('./data/model_1')

step = 10  # 生成初始数据的步长


# filter_jump_step = 5  # 在过滤的时候，跳跃的步长


def judge_range(before_value, shock_range_value):
    if len(shock_range_value) < 60:
        return False

    tmp_max = np.max(shock_range_value)
    if tmp_max < 400:  # 800
        return False

    if before_value / tmp_max > 0.2:
        return False

    after_left = int(max(10, len(shock_range_value) / 8))
    if before_value / np.mean(shock_range_value[:after_left]) > 0.4:
        return False

    return True


def range_filter(shock_value, left, right):
    # 根据形状进行分类
    feature = build_model_1.build_feature(shock_value, left, right)
    if feature is None:
        return False

    # 根据不同的类型分别进行处理
    predict_ret = model_1.predict([feature])[0]
    if predict_ret == 5:
        return filter_5(shock_value, left, right)
    else:
        return [False]


def config():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 7)


def filter_5(shock_value, left, right):
    # 过滤掉非常不靠谱的点
    tmp = shock_value[left:right]
    tmp_max = np.max(tmp)
    tmp_max_index = np.where(tmp == tmp_max)[0][0]
    tmp_min = np.min(tmp[tmp_max_index:tmp_max_index + 10])
    if tmp_max_index <= 2 and (tmp_min / tmp_max > 0.5 or tmp_min / tmp_max < 0.01):
        return [False]
    tmp_index_1 = np.where(tmp > tmp_max * 0.1)[0]
    if len(tmp_index_1) <= 2:
        return [False]

    # 根据峰值过滤
    if tmp_max < 600:
        return [False]

    # 根据最高点附近的区间长度过滤
    before_mean = np.mean(shock_value[left - 10:left])
    a = np.where(tmp[:tmp_max_index] < before_mean)[0]
    b = tmp_max_index
    if len(a) > 0:
        b -= a[-1]
    a = np.where(tmp[tmp_max_index:] < before_mean)[0]
    if len(a) > 0:
        b += a[0]
    else:
        b += len(tmp) - tmp_max_index
    if b < 30:
        return [False]

    # 第一种跳跃点过滤
    first_jump = np.mean(shock_value[left - 20:left]) / (np.mean(shock_value[left:left + 20]) + 1)
    if first_jump > 0.5:
        return [False]

    # 第二种跳跃点进行过滤
    jump_5 = find_second_jump_point(shock_value, left, right, before_mean, 5)
    jump_10 = find_second_jump_point(shock_value, left, right, before_mean, 10)
    jump_point = jump_5.tolist() + jump_10.tolist()
    if len(jump_point) == 0:
        return [False, jump_point]
    else:
        return [True, jump_point]


# 计算跳跃程度
def calc_jump_degree(tmp, s, before_mean, filter_jump_step):
    s = tmp[s:s - filter_jump_step]
    s_mean = np.mean(s.reshape(-1, filter_jump_step), axis=1) + 1
    s_min = np.min(s.reshape(-1, filter_jump_step), axis=1) + 1

    a = (s_mean[:-1] / s_mean[1:])[:-1]
    b = (s_mean[:-2] / s_mean[2:])

    ret = np.min((a, b), axis=0)
    ret[np.where(s_min[:-2] <= max(before_mean * 3, 100))[0]] = -1

    ret[np.where(ret > 0.5)[0]] = -1

    return ret


# 找到全部的跳跃点
def find_second_jump_point(shock_value, left, right, before_mean, filter_jump_step):
    right -= (right - left) % filter_jump_step
    tmp = shock_value[left:right]

    # 找到最后一个满足条件的点（应该这个位置最能反应特征）
    a = np.zeros(int((right - left) / filter_jump_step)) - 1
    for s in np.arange(filter_jump_step):
        a = calc_jump_degree(tmp, s, before_mean, filter_jump_step)
        a[np.where(a != -1)] = s
    b = np.arange(0, len(a) * filter_jump_step, filter_jump_step)

    # 计算真实的位置（只保留前半部分）
    ret = a + b
    ret = ret[np.where(a >= 0)[0]]
    ret = ret[np.where(ret < len(tmp) * (3 / 8))[0]]

    return ret
