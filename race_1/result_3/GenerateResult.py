# 生成最终的结果文件

import json
from obspy import read
import datetime
from matplotlib import pyplot as plt
import numpy as np

INTERVAL = 5
RESULT_FILE = open('./data/result.csv', 'a')
DIR_PATH = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"


def read_filter_result():
    ret = []

    filter_result_file = open('./data/filter_result.txt', 'r')
    while True:
        line = filter_result_file.readline()
        if not line:
            break

        infos = line.split('|')
        lr = json.loads(infos[1])

        ret.append((infos[0], lr[0], lr[1]))
    return ret


def format_time(t):
    return str(float(datetime.datetime.fromtimestamp(t + 8 * 3600).strftime('%Y%m%d%H%M%S.%f')))


def find_p_index(data, left, right):
    # 计算
    tmp = data[left:right].reshape(-1, INTERVAL)
    y = [np.std(tmp, axis=1)][0]
    x = np.arange(len(y))

    limit = max(np.mean(y[:5]) * 2, 150)
    ret = 2

    print("LIMIT = ", limit)

    while ret < len(y):
        if np.mean(y[ret - 2:ret]) > limit:
            break
        ret += 1

    print("RET = ", ret)

    return (ret - 2) * INTERVAL + left

    # 图形化
    # plt.subplot(212)
    # plt.plot(x, y)
    # plt.plot(ret, 0, 'ro')
    # plt.subplot(211)
    # plt.plot(np.arange(right - left), data[left:right])
    # plt.show()


def find_s_index(data, left, right):
    pass


if __name__ == '__main__':
    filter_result = read_filter_result()

    # 生成结果文件
    last_file_name = ''
    delta = None
    data = None
    starttime = None
    location = None

    for record in filter_result:
        filename = record[0]

        if last_file_name != filename:
            file_content = read(DIR_PATH + filename)
            data = file_content[0].data
            delta = file_content[0].stats.delta
            starttime = float(file_content[0].stats.starttime.strftime("%s.%f"))
            location = filename.split('.')[1]
            last_file_name = filename

        # 找到P形波位置
        index = find_p_index(data, record[1], record[2])
        RESULT_FILE.write(location + "," + format_time(starttime + index * delta) + ",P\n")

        # 找到S形波位置
        find_s_index(data, record[1], record[2])
