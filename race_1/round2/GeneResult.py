from obspy import read
import numpy as np
import json
import os
from matplotlib import pyplot as plt
import datetime

DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"
if os.path.exists("./data/result.csv"):
    os.remove("./data/result.csv")
RESULT_FILE = open("./data/result.csv", "w")
INTERVAL = 5


def format_time(t):
    return str(float(datetime.datetime.fromtimestamp(t + 8 * 3600).strftime('%Y%m%d%H%M%S.%f')))


def main():
    feature_file = open('./data/feature.txt')
    last_unit = ""
    value = []
    starttime = 0
    location = ""

    while True:
        line = feature_file.readline()
        if not line:
            break

        # 解析内容。
        infos = line.split('|')
        unit = infos[0]
        lr = json.loads(infos[1])
        left, right = int(lr[0] / INTERVAL), int(lr[1] / 5)

        # 读取文件内容。
        if unit != last_unit:
            last_unit = unit

            file_names = [DIR_PATH + unit + ".BHE", DIR_PATH + unit + ".BHN", DIR_PATH + unit + ".BHZ"]
            file_contents = [read(i) for i in file_names]
            file_datas = [i[0].data for i in file_contents]
            file_stds = [[np.std(data.reshape(-1, INTERVAL), axis=1)][0] for data in file_datas]
            value = (np.array([file_stds[0] + file_stds[1] + file_stds[2]]) / 3)[0]

            starttime = float(file_contents[0][0].stats.starttime.strftime("%s.%f"))
            location = unit.split('.')[1]

        # 搜索数据。
        ret = left
        value_limit = max(np.max(value[left:right]) * 0.01,
                          np.mean(value[left:left + int((right - left) * 0.05)]) * 3)
        print(value_limit)
        while ret < right - INTERVAL * 3:
            a = np.mean(value[ret + INTERVAL * 0:ret + INTERVAL * 1])
            b = np.mean(value[ret + INTERVAL * 1:ret + INTERVAL * 3])
            if a > value_limit and b > value_limit:
                break
            ret += 1

        y = value[left:right]
        x = np.arange(right - left)
        plt.subplot(311)
        plt.plot(x, y)
        # ret += 2
        #
        # RESULT_FILE.write(location + "," + format_time(starttime + ret * INTERVAL * 0.01) + ",P\n")
        # RESULT_FILE.flush()
        #
        # plt.axhline(y=value_limit, color='b')
        # plt.axvline(x=ret - left, color='b')
        y1 = np.mean(y[:len(y) - len(y) % 5].reshape(-1, INTERVAL), axis=1)
        x1 = np.arange(len(y1))
        plt.subplot(312)
        plt.plot(x1, y1)

        y2 = np.mean(y1[:len(y1) - len(y1) % 5].reshape(-1, INTERVAL), axis=1)
        x2 = np.arange(len(y2))
        plt.subplot(313)
        plt.plot(x2, y2)

        plt.show()


if __name__ == '__main__':
    main()

    RESULT_FILE.close()
