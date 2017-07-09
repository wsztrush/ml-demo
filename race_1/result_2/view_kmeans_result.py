from sklearn.externals import joblib
import json
import numpy as np
from obspy import read
from matplotlib import pyplot as plt
from matplotlib import animation

dir_path = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"


# 读取特征文件
def read_feature_file():
    result = []
    feature_file = open("./data/feature.txt")

    while 1:
        line = feature_file.readline()
        if not line:
            break
        value = json.loads(line)
        result.append(value)

    return result


# 读取宽度的数据
def get_interval_vibration(data, interval):
    l = int(len(data) / interval)

    result = np.zeros(l)
    for i in np.arange(l):
        result[i] = np.std(data[i * interval:(i + 1) * interval])

    return result


if __name__ == "__main__":
    # 加载特征
    feature_list = read_feature_file()

    # 加载模型
    kmeans = joblib.load('./data/kmeans')

    # 动画展示聚类的效果
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.set_ylim(0, 10000)
    ax1.set_xlim(0, 10)

    ax2 = fig.add_subplot(212)
    ax2.set_ylim(0, 10000)
    ax2.set_xlim(0, 10)

    line1, = ax1.plot([], [])
    line2, = ax2.plot([], [])

    x = np.arange(10)


    # 获取下个数据
    def next_value():
        last_file_name = ""
        data = []

        for feature in feature_list:
            c = kmeans.predict([feature[3]])
            left, right = feature[1], feature[2]
            if c != 4:
                continue

            file_name = feature[0]

            if file_name != last_file_name:
                last_file_name = file_name
                data = read(dir_path + feature[0])[0].data

            print(last_file_name)

            original_data = data[left:right]
            vibration_data = get_interval_vibration(original_data, 10)

            yield original_data, vibration_data


    # 刷新页面展示
    def refresh(value):
        line1.set_data(np.arange(len(value[0])), value[0])
        line2.set_data(np.arange(len(value[1])), value[1])

        ax1.set_ylim(np.min(value[0]), np.max(value[0]))
        ax1.set_xlim(0, len(value[0]))

        ax2.set_ylim(np.min(value[1]), np.max(value[1]))
        ax2.set_xlim(0, len(value[1]))

        return line1, line2


    # 动画展示
    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=50, repeat=False)
    plt.show()
