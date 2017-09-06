import matplotlib
import numpy as np

origin_dir_path = "/Users/tianchi.gzt/Downloads/race_1/after/"
shock_path = "./data/shock/"
range_path = "./data/range/"

step = 10


def judge_range(before_value, shock_range_value):
    if len(shock_range_value) < 80:
        return False

    tmp_max = np.max(shock_range_value)
    if tmp_max < 800:
        return False

    if before_value / tmp_max > 0.2:
        return False

    after_left = int(max(10, len(shock_range_value) * 0.1))
    if before_value / np.mean(shock_range_value[:after_left]) > 0.6:
        return False

    return True


def config():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 8)
