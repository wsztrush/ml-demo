# 根据范围数据生成最终的特征数据
import time
import numpy as np
import json
from obspy import read
import math

dir_path = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"
feature_file = open("./data/feature.txt", "w")
feature_size = 10


# 读取可能的范围文件
def read_range_file():
    result = {}
    range_file = open('./data/range.txt')

    while 1:
        line = range_file.readline()
        if not line:
            break
        kv = line.split('|')
        result[kv[0]] = json.loads(kv[1])
    return result


# 获取比例
def get_ratio():
    # result = np.logspace(0, 10, 10, base=1.2)
    # result = result / np.sum(result)

    result = (np.zeros(feature_size) + 1) / feature_size
    return result


# 分割比例
ratio = get_ratio()


# 获取均值
def get_original_mean(data):
    result = np.zeros(feature_size + 1)

    index = 0
    for j, ratio_value in enumerate(ratio):
        next_index = int(index + ratio_value * len(data))
        result[j] = np.mean(data[index:next_index])
        index = next_index + 1

    return result


# 获取振幅
def get_vibration(data, interval):
    l = int(len(data) / interval)

    result = np.zeros(l)
    for i in np.arange(l):
        result[i] = np.std(data[i * interval:(i + 1) * interval])

    return result


# 获取振幅的均值
def get_vibration_mean(data):
    result = np.zeros(feature_size + 1)
    ratio = get_ratio()

    index = 0
    for j, ratio_value in enumerate(ratio):
        next_index = int(index + ratio_value * len(data))
        result[j] = np.mean(get_vibration(data[index:next_index], 10))
        index = next_index + 1

    m = np.max(result) + 10

    result /= m
    result[10] = np.log10(m)

    for i in result:
        if math.isnan(i):
            print(result)
            break

    return result


# 获取第一小段的震动幅度
def get_first_vibration(data):
    result = np.zeros(1)
    ratio = 1 / 9


# 处理文件中的范围，生成特征值
def process(key, range_list):
    data = np.array(read(dir_path + key)[0].data)

    result = []
    for i in range_list:
        left, right = max(0, i[0] - int((i[1] - i[0]) / 9)), i[1]
        f = [key, left, right]
        f.append(get_vibration_mean(data[f[1]: f[2]]).tolist())
        f.append()

        result.append(f)
    return result


if __name__ == '__main__':
    # 读取文件
    range_dict = read_range_file()

    # 处理各个文件，生成特征
    feature_list = []
    for k, v in range_dict.items():
        start = time.time()
        for i in process(k, v):
            feature_file.write(json.dumps(i) + "\n")

            # print(k, "%4.2f" % (time.time() - start))
