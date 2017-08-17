# 展示Kmeans分类的结果
#
# 1. 读取特征文件
# 2. 加载模型
# 3. 图形化展示

from sklearn.externals import joblib
import json
import numpy as np
from obspy import read
from matplotlib import pyplot as plt
from matplotlib import animation

DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"
INTERVAL = 5


def get_feature_2(kv):
    return json.loads(kv[3])[:1]


def get_feature_list():
    result = []
    feature_file = open("./data/feature.txt")

    while 1:
        line = feature_file.readline()
        if not line:
            break
        result.append(line)

    return result


if __name__ == '__main__':
    # 加载特征文件
    feature_list = get_feature_list()

    # 加载模型文件
    kmeans = joblib.load('./data/kmeans')

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


    def next_value():
        last_file_name = ""
        data = []
        count = 0

        for f in feature_list:
            # 解析特征文件
            infos = f.split('|')
            filename = infos[0]
            lr = json.loads(infos[1])
            feature = get_feature_2(infos)

            # 使用模型预测，不正常的类型
            c = kmeans.predict([feature])
            if c != 1:
                continue

            count += 1
            print(count)

            # 判断文件名是否相同，不相同重新加载数据
            if filename != last_file_name:
                last_file_name = filename
                data = read(DIR_PATH + filename)[0].data

            # 构造展示数据
            left, right = lr[0], lr[1]
            _origin_data = data[left:right]

            _vibration_data = data[left:right].reshape(-1, INTERVAL)
            _vibration_data = [np.std(_vibration_data, axis=1)][0]

            yield _origin_data, _vibration_data


    def refresh(value):
        line1.set_data(np.arange(len(value[0])), value[0])
        line2.set_data(np.arange(len(value[1])), value[1])

        ax1.set_ylim(np.min(value[0]), np.max(value[0]))
        ax1.set_xlim(0, len(value[0]))

        ax2.set_ylim(np.min(value[1]), np.max(value[1]))
        ax2.set_xlim(0, len(value[1]))

        return line1, line2


    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=500, repeat=False)
    plt.show()
