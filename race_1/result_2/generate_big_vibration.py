# 找到震动幅度骄傲的的位置，方便以后找数据，相当于是建索引吧。
import os
import numpy as np
import json
import time
from obspy import read

step = 5
interval = 1000
vibration_limit = 600

result = open("./data/big_vibration.txt", "a")


def process(dir_path, file_path):
    print(file_path)
    start_time = time.time()

    content = read(dir_path + file_path)
    point_list = []

    data = content[0].data
    l = len(data)

    i = 0
    while i < l:
        vibration = np.std(data[i:i + step])

        if vibration > vibration_limit:
            print(i, vibration)
            point_list.append(i)
            i += interval
        else:
            i += step

    if len(point_list) > 0:
        result.write(file_path + "|" + json.dumps(point_list) + "\n")
        result.flush()

    print("[LEN]", len(point_list), "[RET]",point_list)
    print('[COST] ' + str(time.time() - start_time))


if __name__ == '__main__':
    dir_path = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"
    files = os.listdir(dir_path)

    for file in files:
        process(dir_path, file)
