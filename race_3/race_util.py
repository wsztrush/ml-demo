import matplotlib
import numpy as np
import build_model_1

from sklearn.externals import joblib

origin_dir_path = "/Users/tianchi.gzt/Downloads/race_1/after/"
shock_path = "./data/shock/"
range_path = "./data/range/"

model_1 = joblib.load('./data/model_1')

step = 10


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
    feature = build_model_1.build_feature(shock_value, left, right)

    if feature is None:
        return False

    predict_ret = model_1.predict([feature])[0]

    if predict_ret == 0:
        return False  # TODO
    elif predict_ret == 1:
        return True  # TODO
    elif predict_ret == 2:
        return False
    elif predict_ret == 3:
        return True  # TODO
    elif predict_ret == 4:
        # return filter_4(shock_value, left, right)
        return False
    elif predict_ret == 5:
        return True  # TODO
    elif predict_ret == 6:
        return False

    return False


def config():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 7)


def filter_4(shock_value, left, right):
    # 过滤掉非常不靠谱的点
    tmp = shock_value[left:right]
    tmp_max = np.max(tmp)
    tmp_max_index = np.where(tmp == tmp_max)[0][0]
    tmp_min = np.min(tmp[tmp_max_index:tmp_max_index + 10])
    if tmp_max_index <= 2 and (tmp_min / tmp_max > 0.5 or tmp_min / tmp_max < 0.01):
        return False
    tmp_index_1 = np.where(tmp > tmp_max * 0.1)[0]
    if len(tmp_index_1) <= 2:
        return False

    # 根据峰值过滤
    if tmp_max < 600:
        return False

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
        return False

    return True


def filter_1(shock_value, left, right):
    # 过滤明显不正确的点
    before_mean = np.mean(shock_value[left - 10:left])
    if before_mean > 2000 and left > 100 and shock_value[left - 100] > 50000:
        return False

    # 在最高点附近找区间，根据长度过滤
    tmp = shock_value[left:right]
    tmp_max = np.max(tmp)
    tmp_max_index = np.where(tmp == tmp_max)[0][0]

    a = np.where(tmp[:tmp_max_index] < before_mean)[0]
    b = tmp_max_index
    if len(a) > 0:
        b -= a[-1]
    a = np.where(tmp[tmp_max_index:] < before_mean)[0]
    if len(a) > 0:
        b += a[0]
    else:
        b += len(tmp) - tmp_max_index
    if b < 20:
        return False

    # 根据刚开始的高度过滤
    a = np.mean(tmp[:int(len(tmp) / 10)])
    if a / tmp_max < 0.01:
        return False

    return True  # TODO
