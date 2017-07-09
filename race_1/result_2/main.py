from sklearn.cluster import KMeans
from sklearn.externals import joblib
import json

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
        result.append(value[3])

    return result


if __name__ == "__main__":
    # 加载特征
    feature_list = read_feature_file()

    # 加载模型
    kmeans = joblib.load('./data/kmeans')
