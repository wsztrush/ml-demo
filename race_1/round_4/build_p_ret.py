import os
import datetime
import numpy as np
import matplotlib
import race_config
import build_clf
import random

from matplotlib import pyplot as plt
from sklearn.externals import joblib
from obspy import read

if os.path.exists("./data/1.csv"):
    os.remove("./data/1.csv")
RESULT_FILE = open("./data/1.csv", "w")


def process(file_std, left, right, unit):
    # 解析
    location, starttime = parse_unit(unit)

    # 第一层过滤
    file_std = file_std[left:right]
    std_max = np.max(file_std)
    tmp = np.where(file_std > std_max * 0.2)
    if len(tmp[0]) == 0:
        return
    r_end = np.min(np.where(file_std > std_max * 0.2))
    std_limit = max(np.mean(file_std[:int((right - left) / 11)]) * 2, std_max * 0.005)

    find_step = 5

    ret = r_end
    while ret > 0:
        if np.mean(file_std[ret:ret + find_step]) <= std_limit:
            break
        ret -= 1

    index = np.where(file_std[ret:ret + 5] > std_limit)[0]
    if len(index) > 0:
        ret += np.min(index) - 1
    else:
        ret += 4

    plt.subplot(311)
    plt.plot(np.arange(len(file_std)), file_std)
    plt.axvline(x=r_end, color='g')
    plt.axvline(x=ret, color='r')

    plt.subplot(312)
    plt.plot(np.arange(r_end), file_std[:r_end])
    plt.axvline(x=ret, color='r')
    plt.axvline(x=ret - 8, color='g')
    plt.axvline(x=ret + 8, color='g')
    plt.axhline(y=std_limit, color='orange')

    r_end += find_step - r_end % find_step
    tmp = file_std[:r_end]
    tmp = tmp.reshape(-1, find_step)
    tmp = np.mean(tmp, axis=1)

    plt.subplot(313)
    plt.plot(np.arange(len(tmp)), tmp)
    plt.axhline(y=std_limit, color='r')

    plt.show()

    RESULT_FILE.write(location + "," + format_time(starttime + (ret + left) * 5 * 0.01) + ",P\n")
    RESULT_FILE.flush()


def main():
    unit_list = os.listdir(race_config.RANGE_PATH)
    # random.shuffle(unit_list)

    for unit in unit_list:
        file_std = np.load(race_config.STD_PATH + unit)
        file_std = file_std[2]
        file_range = np.load(race_config.RANGE_PATH + unit)

        for lr in file_range:
            left, right = lr[0], lr[1]
            process(file_std, left, right, unit)


def format_time(t):
    return str(float(datetime.datetime.fromtimestamp(t + 8 * 3600).strftime('%Y%m%d%H%M%S.%f')))


def parse_unit(unit):
    infos = unit.split('.')
    days = infos[2][4:7]

    return infos[1], 1214841600.0 + (int(days) - 183) * 24 * 60 * 60


def check():
    dir_path = "/Users/tianchi.gzt/Downloads/race_1/after/"

    for unit in os.listdir(dir_path):
        file_content = read(dir_path + unit)

        print(float(file_content[0].stats.starttime.strftime("%s.%f")) - parse_unit(unit)[1])


if __name__ == '__main__':
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 8)

    # main()
    check()
