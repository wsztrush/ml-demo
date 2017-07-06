# 文件名|可能的位置列表
#
#
import os
import numpy as np
import json
from obspy import read

step = 5
interval = 1000
vibration_limit = 500

result = open("./data/big_vibration.txt", "a")


def process(dir, file):
    content = read(dir + file)
    point_list = []

    data = content[0].data
    l = len(data)

    i = 0
    while i < l:
        vibration = np.std(data[i:i + step])

        if vibration > vibration_limit:
            point_list.append(i)

            i += interval
        else:
            i += 1

        if len(point_list) > 0:
            result.append(file + "|" + json.dumps(point_list) + "\n")


if __name__ == '__main__':
    dir = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"
    files = os.listdir(dir)

    for file in files:
        process(dir, file)
