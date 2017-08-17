import json
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
from matplotlib import pyplot as plt


def get_feature_2(kv):
    feature = json.loads(kv[3])[:1]

    if np.isnan(feature).any() or not np.isfinite(feature).all():
        print(feature)

    return feature


if __name__ == '__main__':
    feature_file = open("./data/feature.txt", "r")
    feature_list = []

    while 1:
        line = feature_file.readline()
        if not line:
            break
        kv = line.split('|')

        feature_list.append(get_feature_2(kv))

    # 训练
    kmeans = KMeans(n_clusters=5).fit(feature_list)

    print(kmeans.labels_)
    print(np.bincount(kmeans.labels_))
    tmp = kmeans.cluster_centers_
    print(tmp)

    plt.plot(tmp.T)
    plt.show()
    # 保存模型
    joblib.dump(kmeans, './data/kmeans')
