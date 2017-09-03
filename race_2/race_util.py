import matplotlib
import numpy as np

origin_dir_path = "/Users/tianchi.gzt/Downloads/race_1/after/"
shock_path = "./data/shock/"
range_path = "./data/range/"

shock_step = 10


def load_shock_value(unit):
    shock_value_list = np.load(shock_path + unit)
    shock_value = np.sqrt(np.square(shock_value_list[0]) + np.square(shock_value_list[1]) + np.square(shock_value_list[2]))

    return shock_value


def config():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 8)


def get_before_left(left, right):
    return max(int(left - (right - left) * 0.1), 0)
