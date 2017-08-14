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
import time
import multiprocessing

INTERVAL = 5
VIBRATION_LIMIT = 600
RESULT_FILE = open("./data/range.txt", "w")
L = multiprocessing.Lock()
DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"


def judge(tmp_mean, l, r, tmp_limit):
    value = np.mean(tmp_mean[l:r])
    return value > tmp_limit


def process_file(filename):
    start = time.time()

    # 读取文件
    file_content = read(DIR_PATH + filename)
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
        # 计算振幅
        if tmp_mean[index] < VIBRATION_LIMIT:
            index += 1
            continue

        # 查找左、右边界
        left, right = index - INTERVAL, index + INTERVAL
        while left > last_index and judge(tmp_mean, left, left + INTERVAL, tmp_limit):
            left -= INTERVAL

        while right < tmp_len and judge(tmp_mean, right - INTERVAL, right, tmp_limit):
            right += INTERVAL

        # 过滤并保存结果
        if (right - left) * INTERVAL > 1000:
            result_left = max(left + INTERVAL, 0) * INTERVAL
            result_right = (right - INTERVAL) * INTERVAL
            print(result_left, result_right)
            result.append((result_left, result_right))

        # 更新搜索坐标
        last_index = right
        index = last_index + 1

    # 将范围数据写入文件
    print(filename, len(result))
    L.acquire()
    RESULT_FILE.write(filename + "|" + json.dumps(result) + "\n")
    RESULT_FILE.flush()
    L.release()

    # 打印耗时
    print(filename, "COST %2.3f" % (time.time() - start))


if __name__ == '__main__':
    # 找到所有文件
    filename_list = os.listdir(DIR_PATH)

    # 创建线程池并提交任务执行。
    params = []
    pool = multiprocessing.Pool(processes=4)
    pool.map(process_file, filename_list)
