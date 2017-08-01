# 查看生成的范围数据。

from matplotlib import pyplot as plt
from obspy import read
import numpy as np
import json

DIR_PATH = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"
INTERVAL = 5


def process(filename, range_list):
    if len(range_list) == 0:
        return

    # 读取文件
    file_content = read(DIR_PATH + filename)
    data = file_content[0].data

    # 展示内容
    for r in range_list:
        l, r = r[0], r[1]

        l = max(0, int(l - (r - l) * 0.2))
        l = l - l % 5

        print(filename, l, r)

        plt.subplot(211)
        plt.plot(np.arange(r - l), data[l:r])

        tmp = data[l:r].reshape(-1, INTERVAL)
        y = [np.std(tmp, axis=1)][0]
        x = np.arange(len(y))

        plt.subplot(212)
        plt.plot(x, y)

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
        # process(kv[0], json.loads(kv[1]))

    print(total)
