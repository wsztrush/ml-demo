import matplotlib
import numpy as np
import build_model_1

from sklearn.externals import joblib

origin_dir_path = "/Users/tianchi.gzt/Downloads/race_1/after/"
model_1 = joblib.load('./data/model_1')
step = 10  # 生成初始数据的步长


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


def range_filter(shock_value, shock_z_value, left, right):
    # 根据形状进行分类
    feature = build_model_1.build_feature(shock_value, left, right)
    if feature is None:
        return False

    # 根据不同的类型分别进行处理
    predict_ret = model_1.predict([feature])[0]
    if predict_ret == 0:
        return False
        # return filter_0(shock_value, shock_z_value, left, right)
    elif predict_ret == 1:
        return False
        # return filter_1(shock_value, left, right)
    elif predict_ret == 2:
        # return filter_2(shock_value, shock_z_value, left, right)
        return False
    elif predict_ret == 3:
        # return filter_3(shock_value, shock_z_value, left, right)
        return False
    elif predict_ret == 4:
        # return filter_4(shock_value, left, right)
        return False
    elif predict_ret == 5:
        # return filter_5(shock_value, left, right)
        return False
    elif predict_ret == 6:
        return filter_6(shock_value, shock_z_value, left, right)
    else:
        return False


def config():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 7)


def filter_0(shock_value, shock_z_value, left, right):
    return [False]


def filter_1(shock_value, left, right):
    # 该形状的波形可能性比较小，最后考虑。
    return [False]


def filter_2(shock_value, shock_z_value, left, right):
    before_mean = np.mean(shock_value[left - 10:left])

    # 根据跳跃点进行过滤
    jump_5 = find_jump_point(shock_value, left, right, before_mean, 5)
    jump_10 = find_jump_point(shock_value, left, right, before_mean, 10)
    jump_point = jump_5.tolist() + jump_10.tolist()
    if len(jump_point) == 0:
        return [False, jump_point]

    # 根据开始位置的跳跃情况进行过滤
    a = np.mean(shock_value[left - 20:left]) / (np.mean(shock_value[left:left + 20]) + 1)
    b = np.mean(shock_z_value[left - 20:left]) / (np.mean(shock_z_value[left:left + 20]) + 1)

    if a > 0.3 and b > 0.3:
        return [False]

    return [True]


def filter_3(shock_value, shock_z_value, left, right):
    before_mean = np.mean(shock_value[left - 10:left])

    # 根据开始位置的跳跃情况进行过滤
    a = np.mean(shock_value[left - 20:left]) / (np.mean(shock_value[left:left + 20]) + 1)
    b = np.mean(shock_z_value[left - 20:left]) / (np.mean(shock_z_value[left:left + 20]) + 1)

    if a > 0.3 and b > 0.3:
        return [False]

    return [True]


def filter_4(shock_value, left, right):
    return [True]


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
    jump_5 = find_jump_point(shock_value, left, right, before_mean, 5)
    jump_5 = jump_5[np.where(jump_5 < len(tmp) * (3 / 8))[0]]

    jump_10 = find_jump_point(shock_value, left, right, before_mean, 10)
    jump_10 = jump_10[np.where(jump_10 < len(tmp) * (3 / 8))[0]]

    jump_point = jump_5.tolist() + jump_10.tolist()
    if len(jump_point) == 0:
        return [False, jump_point]
    else:
        return [True, jump_point]


# 5322
def filter_6(shock_value, shock_z_value, left, right):
    before_mean = np.mean(shock_value[left - 10:left])

    # 根据刚开始的跳跃程度进行过滤
    a, b = calc_begin_jump_degree(shock_value, shock_z_value, left)
    if a > 0.3 and b > 0.3:
        return False

    # 根据最大值进行过滤
    a = np.max(shock_value[left:right])
    if a < 600:
        return False

    # 根据最大值所在位置的区间宽度进行过滤
    max_index = np.argmax(shock_value[left:right])
    a = calc_max_length(before_mean, shock_value, left, right, max_index)
    if a < 40:
        return False

    # 根据跳跃点进行过滤
    j_5 = find_jump_point(shock_value, left, right, 0, 5, jump_degree_limit=0.3)
    j_10 = find_jump_point(shock_value, left, right, 0, 10, jump_degree_limit=0.3)
    j_5 = j_5[np.where(j_5 > (right - left) * (2 / 8))[0]]
    j_10 = j_10[np.where(j_10 > (right - left) * (2 / 8))[0]]
    if len(j_5) == 0 and len(j_10) == 0:
        return False

    return True


# 计算跳跃程度
def calc_jump_degree(tmp, s, min_limit, filter_jump_step, jump_degree_limit=0.5):
    s = tmp[s:s - filter_jump_step]
    s_mean = np.mean(s.reshape(-1, filter_jump_step), axis=1) + 1
    s_min = np.min(s.reshape(-1, filter_jump_step), axis=1) + 1

    a = (s_mean[:-1] / s_mean[1:])[:-1]
    b = (s_mean[:-2] / s_mean[2:])

    ret = np.min((a, b), axis=0)
    ret[np.where(s_min[:-2] <= min_limit)[0]] = -1  # TODO max(before_mean * 3, 100)

    ret[np.where(ret > jump_degree_limit)[0]] = -1

    return ret


# 找到全部的跳跃点
def find_jump_point(shock_value, left, right, min_limit, filter_jump_step, jump_degree_limit=0.5):
    right -= (right - left) % filter_jump_step
    tmp = shock_value[left:right]

    # 找到最后一个满足条件的点（应该这个位置最能反应特征）
    a = np.zeros(int((right - left) / filter_jump_step)) - 1
    for s in np.arange(filter_jump_step):
        a = calc_jump_degree(tmp, s, min_limit, filter_jump_step, jump_degree_limit=jump_degree_limit)
        a[np.where(a != -1)] = s
    b = np.arange(0, len(a) * filter_jump_step, filter_jump_step)

    # 计算真实的位置
    ret = a + b
    ret = ret[np.where(a >= 0)[0]]
    return ret


# 计算刚开始的跳跃点
def calc_begin_jump_degree(shock_value, shock_z_value, left):
    return np.mean(shock_value[left - 10:left]) / (np.mean(shock_value[left:left + 20]) + 1), np.mean(shock_z_value[left - 10:left]) / (np.mean(shock_z_value[left:left + 20]) + 1)


# 找到最大值对应的下标
def find_max_index(shock_value, left, right):
    tmp = shock_value[left:right]
    return np.where(tmp == np.max(tmp))[0][0] + left


# 计算最高点附近的长度
def calc_max_length(before_mean, shock_value, left, right, max_index):
    # 截取相关的区间
    tmp = shock_value[left:right]

    # 计算前面的长度
    a = np.where(tmp[:max_index] < before_mean)[0]
    b = max_index
    if len(a) > 0:
        b -= a[-1]

    # 计算后面的长度
    a = np.where(tmp[max_index:] < before_mean)[0]
    if len(a) > 0:
        b += a[0]
    else:
        b += len(tmp) - max_index

    return b
