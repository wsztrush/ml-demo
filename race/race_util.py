import matplotlib
import numpy as np
import build_model_1

from sklearn.externals import joblib

origin_dir_path = "/Users/tianchi.gzt/Downloads/race_1/after/"
model_1 = joblib.load('./data/model_1')
step = 10  # 生成初始数据的步长

# 调试信息
debug_jump_point = None


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
    predict_ret = model_1.predict([feature])[0]

    tmp = shock_value[left:right]
    # 开始点的跳跃程度过滤
    a, b = calc_begin_jump_degree(shock_value, shock_z_value, left)
    if b > 0.3 and a > 0.3:
        return False

    # 最大值过滤
    tmp_max = np.max(tmp)
    if tmp_max < 500:
        return False

    # 最大值对应的区间长度过滤
    tmp_max_length = calc_max_length(np.mean(shock_value[left - 10:left]), shock_value, left, right, np.argmax(tmp))
    if tmp_max_length < 20:
        return False

    # 根据最大值位置进行过滤
    tmp_max_index = np.argmax(tmp)
    a = np.min(tmp[tmp_max_index:tmp_max_index + 20])
    if tmp_max_index <= 5 and a / tmp_max > 0.5:
        return False

    # TODO 需要继续添加通用化过滤

    # 个性化过滤
    if predict_ret == 2:
        # return filter_2(shock_value, shock_z_value, left, right)
        return False
    elif predict_ret == 4:
        # return filter_4(shock_value, shock_z_value, left, right)
        return False
    elif predict_ret == 5:
        # return filter_5(shock_value, shock_z_value, left, right)
        return False
    elif predict_ret == 6:
        return filter_6(shock_value, shock_z_value, left, right)
        # return False
    else:
        return False

        # if predict_ret == 0:
        #     return filter_0(shock_value, shock_z_value, left, right)
        # elif predict_ret == 1:
        #     return filter_1(shock_value, shock_z_value, left, right)
        # elif predict_ret == 3:
        #     return filter_3(shock_value, shock_z_value, left, right)


def config():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 7)


#
def filter_0(shock_value, shock_z_value, left, right):
    return False


#
def filter_1(shock_value, shock_z_value, left, right):
    return False


# 8949
def filter_2(shock_value, shock_z_value, left, right):
    before_mean = np.mean(shock_value[left - 10:left])

    # 根据跳跃点过滤
    filter_jump_step = max(int((right - left) * (1 / 16)), 5)
    a = find_jump_point(shock_value, left, right, filter_jump_step, jump_degree_limit=0.3)
    a = a[np.where((a > (right - left) * (1 / 16)) & (a < (right - left) * (3 / 8)))[0]]
    global debug_jump_point
    debug_jump_point = a
    if len(a) == 0:
        return False

    # 根据开头的震动幅度进行过滤
    s_z_before_mean = np.mean(shock_z_value[left - 20:left])
    s_z_after_mean = np.mean(shock_z_value[left:left + 20])
    if s_z_before_mean / s_z_after_mean > 0.25:
        return False

    return True


def filter_3(shock_value, shock_z_value, left, right):
    before_mean = np.mean(shock_value[left - 10:left])
    # TODO
    return False


# 9461
def filter_4(shock_value, shock_z_value, left, right):
    before_mean = np.mean(shock_value[left - 10:left])

    # 根据跳跃点进行过滤
    filter_jump_step = int((right - left) * (1 / 12))
    a = find_jump_point(shock_value, left, right, filter_jump_step, jump_degree_limit=0.3)
    a = a[np.where(a < (right - left) * (4 / 8))[0]]
    global debug_jump_point
    debug_jump_point = a

    if len(a) == 0:
        return False

    # 根据震动过滤
    s_z_before_mean = np.mean(shock_z_value[left - 20:left])
    s_z_mean = calc_shock_mean(shock_z_value, left, right)
    if s_z_before_mean / s_z_mean[0] > 0.3 or s_z_before_mean / s_z_mean[1] > 0.3:
        return False

    return True


# 10248
def filter_5(shock_value, shock_z_value, left, right):
    tmp = shock_value[left:right]
    before_mean = np.mean(shock_value[left - 10:left])

    # 过滤掉非常不靠谱的点
    tmp_max = np.max(tmp)
    tmp_max_index = np.argmax(tmp)
    tmp_min = np.min(tmp[tmp_max_index:tmp_max_index + 10])
    if tmp_max_index <= 2 and (tmp_min / tmp_max > 0.5 or tmp_min / tmp_max < 0.01):
        return False
    tmp_index_1 = np.where(tmp > tmp_max * 0.1)[0]
    if len(tmp_index_1) <= 2:
        return False

    # 根据跳跃点进行过滤
    jump_step = max(int((right - left) * (1 / 8 / 3)), 5)
    # print(jump_step)
    a = find_jump_point(shock_value, left, right, jump_step, min_limit=max(before_mean * 3, tmp_max * 0.01), jump_degree_limit=0.4)
    a = a[np.where(a < (right - left) * (3 / 8))]
    global debug_jump_point
    debug_jump_point = a
    if len(a) == 0:
        return False

    return True


# 5840：还需要进一步优化
def filter_6(shock_value, shock_z_value, left, right):
    s_max = np.max(shock_value[left:right])

    # 中间比较低的地方就不要了
    s_z_mean = calc_shock_mean(shock_z_value, left, right)
    s_z_before_mean = np.mean(shock_z_value[left - 20:left])
    if s_z_before_mean / s_z_mean[1] > 0.5 or s_z_before_mean / s_z_mean[2] > 0.5:
        return False

    # 第一二段应该至少有一个是比较平静的，如果波动比较大，那么可能不大正常。
    s_mean = calc_shock_mean(shock_value, left, right)
    s_std = calc_shock_std(shock_value, left, right)
    if min(s_std[1] / s_mean[1], s_std[2] / s_mean[2]) > 0.5:
        return False

    # 前面小的太过分了，也是不正常的。
    if s_mean[1] / s_max <= 0.02 or s_mean[2] / s_max <= 0.02:
        return False

    # 根据跳跃点进行过滤
    filter_jump_step = int((right - left) * (1 / 12))
    a = find_jump_point(shock_value, left, right, filter_jump_step, jump_degree_limit=0.3)
    a = a[np.where((a < (right - left) * (5 / 8)))[0]]  # ((right - left) * (1 / 8) < a) &
    global debug_jump_point
    debug_jump_point = a
    # if len(a) == 0:
    #     return False

    return True


# 计算跳跃程度
def calc_jump_degree(tmp, s, filter_jump_step, min_limit=0, jump_degree_limit=0.5):
    s_value = tmp[s:s - filter_jump_step]
    s_mean = np.mean(s_value.reshape(-1, filter_jump_step), axis=1) + 1
    s_min = np.min(s_value.reshape(-1, filter_jump_step), axis=1) + 1

    a = (s_mean[:-1] / s_mean[1:])[:-1]
    b = (s_mean[:-2] / s_mean[2:])

    ret = np.min((a, b), axis=0)

    ret[np.where(s_min[:-2] <= min_limit)[0]] = -1
    ret[np.where(ret > jump_degree_limit)[0]] = -1

    return ret


# 找到全部的跳跃点
def find_jump_point(shock_value, left, right, filter_jump_step, jump_degree_limit=0.5, min_limit=0):
    right -= (right - left) % filter_jump_step
    tmp = shock_value[left:right]

    # 找到最后一个满足条件的点（应该这个位置最能反应特征）
    a = np.zeros(int((right - left) / filter_jump_step) + 1) - 1
    for s in np.arange(filter_jump_step):
        b = calc_jump_degree(tmp, s, filter_jump_step, min_limit=min_limit, jump_degree_limit=jump_degree_limit)
        a[np.where(b != -1)] = s
    b = np.arange(0, len(a) * filter_jump_step, filter_jump_step)

    # 计算真实的位置
    ret = a + b
    ret = ret[np.where(a >= 0)[0]]
    return ret


# 计算刚开始的跳跃点
def calc_begin_jump_degree(shock_value, shock_z_value, left):
    return np.mean(shock_value[left - 20:left]) / (np.mean(shock_value[left:left + 20]) + 1), np.mean(shock_z_value[left - 10:left]) / (np.mean(shock_z_value[left:left + 20]) + 1)


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


# 将区间分成8分，分别计算均值
def calc_shock_mean(shock_value, left, right):
    right -= (right - left) % 8
    tmp = shock_value[left:right].reshape(8, -1)
    tmp = np.mean(tmp, axis=1)

    return tmp


# 将区间分成8分，分别计算标准差
def calc_shock_std(shock_value, left, right):
    right -= (right - left) % 8
    tmp = shock_value[left:right].reshape(8, -1)
    tmp = np.std(tmp, axis=1)

    return tmp


# 将区间分成8分，分别计算最大值
def calc_shock_max(shock_value, left, right):
    right -= (right - left) % 8
    tmp = shock_value[left:right].reshape(8, -1)
    tmp = np.max(tmp, axis=1)

    return tmp


# 将区间分成8分，分别计算最小值
def calc_shock_min(shock_value, left, right):
    right -= (right - left) % 8
    tmp = shock_value[left:right].reshape(8, -1)
    tmp = np.min(tmp, axis=1)

    return tmp
