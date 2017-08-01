# 训练Kmeans
#
# 1. 读取特征文件
# 2. 读取特征进行训练

import json
from sklearn.cluster import KMeans
from sklearn.externals import joblib

if __name__ == '__main__':
    feature_file = open("./data/feature.txt", "r")
    feature_list = []

    while 1:
        line = feature_file.readline()
        if not line:
            break
        kv = line.split('|')

        feature_list.append(json.loads(kv[2]))

    # 训练
    kmeans = KMeans(n_clusters=10).fit(feature_list)

    # 保存模型
    joblib.dump(kmeans, './data/kmeans')
