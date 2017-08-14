# 生成特征文件
#
# 1. 读取范围文件
# 2. 读取原始数据文件
# 3. 生成特征，并写入文件，格式为：文件名|范围|特征一|特征二|特征三

from obspy import read
import numpy as np
import json
import multiprocessing

DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"
INTERVAL = 5
L = multiprocessing.Lock()
RESULT_FILE = open("./data/feature.txt", "w")


def process(line):
    kv = line.split('|')
    filename = kv[0]
    range_list = json.loads(kv[1])

    if len(range_list) == 0:
        return

    file_content = read(DIR_PATH + filename)
    data = file_content[0].data

    result = []
    for r in range_list:
        left, right = r[0], r[1]

        left = max(0, int(left - (right - left) * 0.2))
        left = left - left % 5

        print(filename, left, right)

        tmp = data[left:right].reshape(-1, INTERVAL)
        tmp_mean = [np.std(tmp, axis=1)][0]
        tmp_mean_len = len(tmp_mean) - len(tmp_mean) % 10
        tmp_mean = tmp_mean[:tmp_mean_len].reshape(10, -1)

        # 特征一：分成10等分，分别计算与最大值的比例
        _10_mean_feature = [np.mean(tmp_mean, axis=1)][0]
        _10_mean_feature = _10_mean_feature / np.max(_10_mean_feature)

        result.append(filename + "|" + json.dumps((left, right)) + "|" + json.dumps(_10_mean_feature.tolist()))

    # 将特征写出到文件
    L.acquire()
    for r in result:
        RESULT_FILE.write(r + "\n")
    RESULT_FILE.flush()
    L.release()


if __name__ == '__main__':
    range_file = open("./data/range.txt", "r")
    range_file_content = []

    while 1:
        line = range_file.readline()
        if not line:
            break
        range_file_content.append(line)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, range_file_content)
