import os
import datetime
import numpy as np
import matplotlib

from matplotlib import pyplot as plt
from sklearn.externals import joblib
from obspy import read

STD_PATH = "./data/std/"
RANGE_PATH = "./data/range/"
MODEL_FILE = "./data/clf"
D_SIZE = 20

if os.path.exists("./data/result.csv"):
    os.remove("./data/result.csv")
RESULT_FILE = open("./data/result.csv", "w")


def get_x(file_std, left, right):
    right += D_SIZE - (right - left) % D_SIZE

    tmp = file_std[left:right]
    tmp_max = np.max(tmp)
    tmp = tmp.reshape(D_SIZE, -1)
    tmp = np.mean(tmp, axis=1)

    ret = tmp / (tmp_max + 1.0)
    ret = ret.tolist()

    ret.append(right - left)
    ret.append(tmp_max)

    return ret


def process(file_std, left, right, unit):
    # 解析
    location, starttime = parse_unit(unit)

    # 第一层过滤
    file_std = file_std[left:right]
    std_max = np.max(file_std)
    r_end = np.min(np.where(file_std > std_max * 0.15))
    std_limit = np.mean(file_std[:int(r_end * 0.5)]) * 3

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

    # plt.subplot(311)
    # plt.plot(np.arange(len(file_std)), file_std)
    # plt.axvline(x=r_end, color='g')
    # plt.axvline(x=ret, color='r')
    #
    # plt.subplot(312)
    # plt.plot(np.arange(r_end), file_std[:r_end])
    # plt.axvline(x=ret, color='r')
    # plt.axvline(x=ret - 8, color='g')
    # plt.axvline(x=ret + 8, color='g')
    # plt.axhline(y=std_limit, color='orange')
    #
    # r_end += find_step - r_end % find_step
    # tmp = file_std[:r_end]
    # tmp = tmp.reshape(-1, find_step)
    # tmp = np.mean(tmp, axis=1)
    #
    # plt.subplot(313)
    # plt.plot(np.arange(len(tmp)), tmp)
    # plt.axhline(y=std_limit, color='r')
    #
    # plt.show()

    print(location)

    RESULT_FILE.write(location + "," + format_time(starttime + (ret + left) * 5 * 0.01) + ",P\n")
    RESULT_FILE.flush()


def main():
    clf = joblib.load(MODEL_FILE)

    for unit in os.listdir(RANGE_PATH):
        file_std = np.load(STD_PATH + unit)[2]
        file_range = np.load(RANGE_PATH + unit)
        for lr in file_range:
            left, right = lr[0], lr[1]

            x = get_x(file_std, left, right)

            if clf.predict([x]) == 1:
                process(file_std, left, right, unit)


def format_time(t):
    return str(float(datetime.datetime.fromtimestamp(t + 8 * 3600).strftime('%Y%m%d%H%M%S.%f')))


def parse_unit(unit):
    infos = unit.split('.')
    days = infos[2][4:7]

    return infos[1], 1214841600.0 + (int(days) - 183) * 24 * 60 * 60


def test():
    dir_path = "/Users/tianchi.gzt/Downloads/race_1/after/"

    for unit in os.listdir(dir_path):
        file_content = read(dir_path + unit)

        print(float(file_content[0].stats.starttime.strftime("%s.%f")) - parse_unit(unit)[1])


if __name__ == '__main__':
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 8)

    main()
    # test()
