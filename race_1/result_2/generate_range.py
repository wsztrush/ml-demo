import time
import numpy as np
import json
from obspy import read
from matplotlib import pyplot as plt

dir_path = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"
range_file = open("./data/range.txt", "w")


# 读取振幅文件，生成字典
def read_vibration_file():
    result = {}
    big_vibration_file = open('./data/big_vibration.txt')
    while 1:
        line = big_vibration_file.readline()
        if not line:
            break
        kv = line.split('|')
        result[kv[0]] = json.loads(kv[1])

    for value in result.values():
        value.sort()

    return result


# 计算振幅
def get_vibration(data):
    # return np.sum(np.abs(data - np.mean(data))) / len(data)
    return np.std(data)


# 获取根据分割的一系列振幅
def get_interval_vibration(data, interval):
    l = int(len(data) / interval)

    result = np.zeros(l)
    for i in np.arange(l):
        result[i] = get_vibration(data[i * interval:(i + 1) * interval])

    return result


# 获取间隔的平均值
def get_intervl_mean_vibration(data, interval):
    l = int(len(data) / interval)

    result = np.zeros(l)
    for i in np.arange(l):
        result[i] = np.mean(data[i * interval:(i + 1) * interval])
    return result


# 检查一个区间里面是否有震动
def judge_vibration(data, interval, vibration_limit):
    tmp = get_interval_vibration(data, interval)
    # tmp.sort()

    # return tmp[int(len(data) / interval * 0.6)] > 600
    return np.mean(tmp) > vibration_limit


# 展示数据
def show_result(data, left, right, interval):
    left -= 1000

    plt.subplot(311)
    plt.plot(np.arange(right - left), data[left:right])

    # 宽度
    x1 = np.arange(int((right - left) / interval))
    y1 = get_interval_vibration(data[left:right], interval)

    plt.subplot(312)
    plt.plot(x1, y1)

    # 宽度的变化情况
    y2 = get_intervl_mean_vibration(y1, interval)
    x2 = np.arange(len(y2))

    plt.subplot(313)
    plt.plot(x2, y2)

    plt.show()


# 展示原始数据的形状
def show_data(data, value, interval):
    x1 = np.arange(2000)
    y1 = data[value - 1000: value + 1000]
    plt.subplot(311)
    plt.plot(x1, y1)

    x2 = np.arange(int(2000 / interval))
    y2 = get_interval_vibration(y1, interval)
    plt.subplot(312)
    plt.plot(x2, y2)

    x3 = np.arange(int(2000 / interval / interval))
    y3 = get_intervl_mean_vibration(y2, interval)
    plt.subplot(313)
    plt.plot(x3, y3)
    plt.show()


# 读取每个振幅较高的点，获取它可能范围，并构造这个范围内的一些特征
def process(key, values):
    result = []

    data = np.array(read(dir_path + key)[0].data)

    step, interval = 100, 10
    last_index = 0
    l = len(data)

    # 计算平均的震动幅度，作为阈值
    d = data[:l - l % interval].reshape(-1, interval)
    d_std = [np.std(d, axis=1)]
    vibration_limit = max(50, min(300, np.mean(np.frompyfunc(lambda i: 0 if i > 1e6 else i, 1, 1)(d_std))))

    print("vibration_limit = ", "%10.0f" % vibration_limit)

    for value in values:
        left, right = value, value

        if value > last_index:
            while left - step > last_index and judge_vibration(data[left - step:left], interval, vibration_limit):
                left -= step

            while right + step < l and judge_vibration(data[right:right + step], interval, vibration_limit):
                right += step

        last_index = max(last_index, right)

        if right - left < 1000:
            continue

        # 绘制数据
        # show_result(data, left, right, interval)

        # 添加到结果中
        result.append((left, right))

    return result


if __name__ == '__main__':
    # 读取生成好的文件
    vibration_dict = read_vibration_file()

    # 依次处理每个文件
    record_count = 0
    for k, v in vibration_dict.items():
        start = time.time()
        record_list = process(k, v)

        if len(record_list) > 0:
            range_file.write(k + "|" + json.dumps(record_list) + "\n")
            range_file.flush()

        record_count += len(record_list)
        print(k, "%2.3f" % (time.time() - start), len(record_list))

    print("find record counts is " + str(record_count))

    # x = "GS.WXT.2008218000000.BHN"
    # print(vibration_dict[x])
    # process(x, vibration_dict[x])
