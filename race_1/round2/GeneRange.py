import os
import numpy as np
import json
import multiprocessing
import time
from obspy import read

DIR_PATH = '/Users/tianchi.gzt/Downloads/race_1/after/'
RESULT_FILE = open('./data/range.txt', 'a')
INTERVAL = 5
VIBRATION_LIMIT = 500
L = multiprocessing.Lock()


# 查找左边界
def find_left(file_std, index, last_index):
    ret = index - INTERVAL
    while ret > last_index and np.mean(file_std[ret:ret + INTERVAL]) > 100:
        ret -= INTERVAL
    return ret


# 查找右边界
def find_right(file_std, index, index_end):
    ret = index + INTERVAL
    while ret < index_end and np.mean(file_std[ret - INTERVAL:ret]) > 100:
        ret += INTERVAL
    return ret


# 处理单个单元
def process(unit):
    start = time.time()

    # 读取文件内容
    file_names = [DIR_PATH + unit + ".BHE", DIR_PATH + unit + ".BHN", DIR_PATH + unit + ".BHZ"]
    file_conents = [read(i) for i in file_names]
    file_datas = [i[0].data for i in file_conents]

    # 计算震动幅度
    file_stds = [[np.std(data.reshape(-1, INTERVAL), axis=1)][0] for data in file_datas]

    # 查找范围
    index, last_index, index_end = 0, 0, len(file_stds[0])
    result = []
    while index < index_end:
        if np.max([i[index] for i in file_stds]) < VIBRATION_LIMIT:
            index += 1
            continue

        left = np.min([find_left(i, index, last_index) for i in file_stds])
        right = np.max([find_right(i, index, index_end) for i in file_stds])

        if (right - left) * INTERVAL > 300:
            r = (int(max(left + INTERVAL, 0) * INTERVAL), int((right - INTERVAL) * INTERVAL))
            # print(unit + "|" + json.dumps(r))
            result.append(r)

        last_index = right
        index = last_index + 1

    # 写入文件
    print(unit, len(result), time.time() - start)
    L.acquire()
    RESULT_FILE.write(unit + "|" + json.dumps(result) + "\n")
    RESULT_FILE.flush()
    L.release()


def main():
    unit_set = set()
    for filename in os.listdir(DIR_PATH):
        unit_set.add(filename[:-4])

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)


if __name__ == '__main__':
    main()
