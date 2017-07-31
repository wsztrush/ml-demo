# 找到特征位置
#
# 1. 遍历after文件
# 2. 计算长度为step(5)的标准差，作为振幅
# 3. 找到振幅超过阈值vibration_limit(600)的位置，记录下来
# 4. 在找到振幅较大的点之后，做一个长度为interval(1000)的跳跃，因为震动附近震动都比较大，没有必要把所有震动大的地方都记录下来
# 5. 将结果写入文件./data/big_vibration.txt，格式为：文件名|位置数组JSON字符串

import os
import numpy as np
import json
import time
from obspy import read

step = 5
interval = 1000
vibration_limit = 600

result = open("./data/big_vibration.txt", "a")


# 处理单个文件
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

    print("[LEN]", len(point_list), "[RET]", point_list)
    print('[COST] ' + str(time.time() - start_time))


if __name__ == '__main__':
    dir_path = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"
    files = os.listdir(dir_path)

    for file in files:
        process(dir_path, file)
