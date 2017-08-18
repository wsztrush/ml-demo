from matplotlib import pyplot as plt
from obspy import read
import numpy as np
import json

DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"
INTERVAL = 5


def process(unit, range_list):
    if len(range_list) == 0:
        return

    # 读取文件
    file_names = [DIR_PATH + unit + ".BHE", DIR_PATH + unit + ".BHN", DIR_PATH + unit + ".BHZ"]
    file_conents = [read(i) for i in file_names]
    file_datas = [i[0].data for i in file_conents]
    file_stds = [[np.std(data.reshape(-1, INTERVAL), axis=1)][0] for data in file_datas]

    # 展示内容
    for r in range_list:
        left, right = r[0], r[1]

        left = max(0, int(left - (right - left) * 0.2))
        left = left - left % INTERVAL

        plt.subplot(321)
        plt.plot(np.arange(right - left), file_datas[0][left:right])

        plt.subplot(322)
        plt.plot(np.arange(int((right - left) / INTERVAL)), file_stds[0][int(left / INTERVAL):int(right / INTERVAL)])

        plt.subplot(323)
        plt.plot(np.arange(right - left), file_datas[1][left:right])

        plt.subplot(324)
        plt.plot(np.arange(int((right - left) / INTERVAL)), file_stds[1][int(left / INTERVAL):int(right / INTERVAL)])

        plt.subplot(325)
        plt.plot(np.arange(right - left), file_datas[2][left:right])

        plt.subplot(326)
        plt.plot(np.arange(int((right - left) / INTERVAL)), file_stds[2][int(left / INTERVAL):int(right / INTERVAL)])

        plt.show()


if __name__ == '__main__':
    range_file = open("./data/range.txt", "r")

    total = 0
    while 1:
        line = range_file.readline()
        if not line:
            break
        kv = line.split('|')
        total += len(json.loads(kv[1]))
        process(kv[0], json.loads(kv[1]))
    print(total)