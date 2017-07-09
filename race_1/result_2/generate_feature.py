# 根据范围数据生成最终的特征数据
import time
import numpy as np
import json
from obspy import read
from matplotlib import pyplot as plt
from matplotlib import animation

dir_path = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"
feature_file = open("./data/feature.txt", "a")


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
    result = np.logspace(0, 10, 10, base=1.2)
    result = result / np.sum(result)
    return result


# 获取均值
def get_original_mean(data):
    result = np.zeros(10)
    ratio = get_ratio()

    index = 0
    for i, ratio_value in enumerate(ratio):
        next_index = int(index + ratio_value * len(data))
        result[i] = np.mean(data[index:next_index])
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
    result = np.zeros(11)
    ratio = get_ratio()

    index = 0
    for i, ratio_value in enumerate(ratio):
        next_index = int(index + ratio_value * len(data))
        result[i] = np.mean(get_vibration(data[index:next_index], 10))
        index = next_index + 1

    m = np.max(result)
    result /= m
    result[10] = np.log10(m)
    return result


# 处理文件中的范围，生成特征值
def process(key, range_list):
    data = np.array(read(dir_path + key)[0].data)

    # fig = plt.figure()
    #
    # ax = fig.add_subplot(111)
    # ax.set_ylim(0, 10000)
    # ax.set_xlim(0, 10)
    # line, = ax.plot([], [])

    # x = np.arange(10)
    #
    # def next_value():
    #     for i in range_list:
    #         left, right = i[0], i[1]
    #         yield get_vibration_mean(data[left:right])
    #
    # def refresh(next_value):
    #     line_value = next_value
    #     line.set_data(x, line_value)
    #     ax.set_ylim(0, np.max(line_value) * 1.5)
    #     return line

    # 动画展示
    # ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=50, repeat=False)
    # plt.show()

    result = []
    for i in range_list:
        f = [key, i[0] - int((i[1] - i[0]) * 0.2), i[1]]
        f.append(get_vibration_mean(data[f[1]: f[2]]).tolist())

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

        print(k, "%4.2f" % (time.time() - start))
