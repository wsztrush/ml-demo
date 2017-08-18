import json
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from obspy import read

DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"
MODEL_FILE = "./data/kmeans_1"
FILTER_FILE = "./data/filter_1.txt"
INTERVAL = 5


# 训练模型
def kmeans_fit():
    feature_file = open("./data/feature.txt", "r")
    feature_list = []

    # 读取数据。
    while 1:
        line = feature_file.readline()
        if not line:
            break

        feature = [json.loads(i) for i in line.split('|')[2:]]
        if np.isnan(feature).any() or not np.isfinite(feature).all():
            print(line)

        feature_list.append(feature[0])
        feature_list.append(feature[1])
        feature_list.append(feature[2])

    # 训练模型。
    kmeans = KMeans(n_clusters=5).fit(feature_list)
    print(kmeans.labels_)
    print(np.bincount(kmeans.labels_))
    tmp = kmeans.cluster_centers_
    print(tmp)

    plt.plot(tmp.T)
    plt.show()

    # 保存模型
    joblib.dump(kmeans, MODEL_FILE)


# 查看训练完成的结果
def kmeans_view():
    # 加载模型
    kmeans = joblib.load(MODEL_FILE)

    # 构建图形
    fig = plt.figure()

    axs = [fig.add_subplot(321 + i) for i in np.arange(6)]
    for ax in axs:
        ax.set_ylim(0, 10000)
        ax.set_xlim(0, 10)

    lines = [ax.plot([], [])[0] for ax in axs]

    # 构建动画
    def next_value():
        last_unit = ""
        file_datas = []
        file_stds = []

        # 读取特征文件
        feature_file = open("./data/feature.txt")
        while 1:
            file_line = feature_file.readline()
            if not file_line:
                break

            infos = file_line.split('|')
            unit = infos[0]
            lr = json.loads(infos[1])
            left, right = lr[0], lr[1]
            feature = [json.loads(i) for i in infos[2:]]

            # 预测分类
            cs = [kmeans.predict([i])[0] for i in feature]

            if cs[0] != 3 and cs[1] != 3 and cs[2] != 3:
                continue

            # 重新读取数据
            if unit != last_unit:
                file_names = [DIR_PATH + unit + ".BHE", DIR_PATH + unit + ".BHN", DIR_PATH + unit + ".BHZ"]
                file_conents = [read(i) for i in file_names]
                file_datas = [i[0].data for i in file_conents]
                file_stds = [[np.std(data.reshape(-1, INTERVAL), axis=1)][0] for data in file_datas]

            # 返回数据
            yield file_datas[0][left:right], file_stds[0][int(left / 5):int(right / 5)], \
                  file_datas[1][left:right], file_stds[1][int(left / 5):int(right / 5)], \
                  file_datas[2][left:right], file_stds[2][int(left / 5):int(right / 5)]

    def refresh(values):
        for i in np.arange(6):
            lines[i].set_data(np.arange(len(values[i])), values[i])

            axs[i].set_ylim(np.min(values[i]), np.max(values[i]))
            axs[i].set_xlim(0, len(values[i]))

        return lines

    ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=500, repeat=False)
    plt.show()


# 生成过滤文件
def kmeans_filter():
    # 加载模型
    kmeans = joblib.load(MODEL_FILE)

    # 读取文件
    feature_file = open("./data/feature.txt", "r")
    result_file = open(FILTER_FILE, "w")
    while True:
        file_line = feature_file.readline()
        if not file_line:
            break

        infos = file_line.split('|')
        unit = infos[0]
        feature = [json.loads(i) for i in infos[2:]]

        cs = [kmeans.predict([i])[0] for i in feature]
        if cs[0] == 3 and cs[1] == 3 and cs[2] == 3:
            result_file.write(unit + infos[1] + "\n")
            result_file.flush()

    result_file.close()


if __name__ == '__main__':
    kmeans_fit()
    # kmeans_view()
    # kmeans_filter()
