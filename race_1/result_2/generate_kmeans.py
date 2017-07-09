import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib

dir_path = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"


# 读取特征文件
def read_feature_file():
    result = []
    feature_file = open("./data/feature.txt")

    while 1:
        line = feature_file.readline()
        if not line:
            break
        result.append(json.loads(line))

    return result


if __name__ == "__main__":
    # 读取数据
    feature_list = np.array(read_feature_file())

    # 加载模型


    # 构造聚类
    kmeans = KMeans(n_clusters=10, random_state=0).fit(feature_list)

    # 保存模型
    joblib.dump(kmeans, './data/kmeans')
