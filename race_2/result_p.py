import os
import datetime
import numpy as np
import matplotlib
import race_util
import random

from matplotlib import pyplot as plt
from obspy import read

if os.path.exists("./data/1.csv"):
    os.remove("./data/1.csv")
result_file = open("./data/1.csv", "w")


def format_time(t):
    return str(float(datetime.datetime.fromtimestamp(t + 8 * 3600).strftime('%Y%m%d%H%M%S.%f')))


def parse_unit(unit):
    infos = unit.split('.')
    days = infos[2][4:7]

    return infos[1], 1214841600.0 + (int(days) - 183) * 24 * 60 * 60


def process(unit):
    # 解析
    location, starttime = parse_unit(unit)

    # 读取数据
    shock_value = np.load('./data/shock/' + unit)
    origin_value = read(race_util.origin_dir_path + unit[:-4] + '.BHZ')[0].data
    range_value = np.load('./data/range/' + unit)

    result = []
    for left, right in range_value:
        before_left = max(int(left - (right - left) / 9), 0)

        tmp_shock = shock_value[before_left:right]
        shock_max = np.max(tmp_shock)
        shock_right_limt = shock_max * 0.2

        # 找到大于切割点的第一个振幅较大的位置
        index = np.where(tmp_shock > shock_right_limt)[0]
        right_end = 0
        for i in index:
            if i > left - before_left:
                right_end = i + 2
                break

        # 计算前面比较小的振幅
        a_right = right_end - right_end % 5
        if right_end == 0:
            continue

        a = max(np.min(np.mean(tmp_shock[:a_right].reshape(5, -1), axis=1)) * 2, shock_max * 0.01, np.max(tmp_shock[:right_end]) * 0.1)

        # 计算大概的位置
        ret = right_end
        while ret > 0:
            if np.mean(tmp_shock[ret:ret + 5]) < a:
                break
            ret -= 1
        b = np.where(tmp_shock[ret:ret + 5] > a)[0]
        if len(b) > 0:
            ret += b[0]
        else:
            ret += 5

        result.append((before_left + ret) * race_util.shock_step)

        # 展示结果
        plt.subplot(4, 1, 1)
        plt.axhline(y=shock_max * 0.1, color='red', linestyle=":")
        plt.axhline(y=0, color="black")
        plt.axvline(x=right_end, color='red', linestyle=":")
        plt.axvline(x=left - before_left, color='green', linestyle=":")
        plt.plot(np.arange(right - before_left), tmp_shock)

        plt.subplot(4, 1, 2)
        plt.axhline(y=a, color='red', linestyle=":")
        plt.axvline(x=ret, color='green')
        plt.bar(np.arange(right_end), tmp_shock[:right_end])

        plt.subplot(4, 1, 3)
        plt.axvline(x=ret * race_util.shock_step, color='red')
        plt.plot(np.arange((right - before_left) * race_util.shock_step), origin_value[before_left * race_util.shock_step:right * race_util.shock_step])

        plt.subplot(4, 1, 4)
        plt.axvline(x=ret * race_util.shock_step, color='red')
        plt.axvline(x=ret * race_util.shock_step - 40, color='green', linestyle=":")
        plt.axvline(x=ret * race_util.shock_step + 40, color='green', linestyle=":")
        plt.plot(np.arange(right_end * race_util.shock_step), origin_value[before_left * race_util.shock_step:(before_left + right_end) * race_util.shock_step])
        plt.show()

    result = sorted(result)

    tmp = []
    for i in result:
        if len(tmp) > 0 and i - tmp[-1] < 40:
            tmp[-1] = (i + tmp[-1]) / 2
        else:
            tmp.append(i)

    print(unit, len(result), len(tmp))

    if len(tmp) > 0:
        for i in tmp:
            result_file.write(location + "," + format_time(starttime + i * 0.01) + ",P\n")
            result_file.flush()


def main():
    unit_list = os.listdir('./data/range/')
    random.shuffle(unit_list)
    for unit in unit_list:
        print(unit)
        process(unit)


if __name__ == '__main__':
    race_util.config()

    main()
