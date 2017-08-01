# 计算范围
#
# 1. 读取文件
# 2. 依次计算长度为INVENTORY(5)的标准差，作为振幅
# 3. 当发现振幅超过FIRST_LIMIT(600)的位置，开始分别向前、向后查找
# 4. 将结果写入文件

import os
from obspy import read
import numpy as np
import json

INTERVAL = 5
VIBRATION_LIMIT = 600
RESULT_FILE = open("./data/range.txt", "w")


def process_file(dir_path, filename):
    # 读取文件
    file_content = read(dir_path + filename)
    data = file_content[0].data
    date_len = len(data)

    # 计算振幅
    tmp = data[:date_len - date_len % INTERVAL].reshape(-1, INTERVAL)
    tmp_mean = [np.std(tmp, axis=1)][0]
    tmp_limit = max(50, min(300, np.mean(np.frompyfunc(lambda i: 0 if i > 1e6 else i, 1, 1)(tmp_mean))))
    tmp_len = len(tmp_mean)

    # 查找范围
    index = 0
    last_index = 0
    result = []
    while index < tmp_len:
        if tmp_mean[index] < VIBRATION_LIMIT:
            index += 1
            continue

        left = right = index
        badcase = 0

        # 查找左边界
        while --left > last_index and badcase < 2:
            if tmp_mean[left] < tmp_limit:
                badcase += 1
            else:
                badcase = 0

        # 查找右边界
        badcase = 0
        while ++right < tmp_len and badcase < 2:
            if tmp_mean[right] < tmp_limit:
                badcase += 1
            else:
                badcase = 0

        # 基本过滤一下，然后保存结果
        if (right - left) * INTERVAL < 1000:
            continue
        else:
            result.append((left, right))

        # 准备下次搜索
        last_index = max(right, last_index)
        index = last_index + 1

    # 将范围数据写入文件
    print(filename, len(result))
    if len(result) > 0:
        RESULT_FILE.write(filename + "|" + json.dumps(result) + "\n")


if __name__ == '__main__':
    dir_path = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"
    filename_list = os.listdir(dir_path)

    # for filename in filename_list:
    #     process_file(dir_path, filename)
    process_file(dir_path, "XX.PWU.2008229000000.BHE")
