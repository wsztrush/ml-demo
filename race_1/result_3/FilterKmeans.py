# 根据KMeans的结果进行过滤

from sklearn.externals import joblib
import json

RESULT_FILE = open("./data/filter_1.txt", "w")


def get_feature_list():
    result = []
    feature_file = open("./data/feature.txt")

    while 1:
        line = feature_file.readline()
        if not line:
            break
        result.append(line)

    return result


if __name__ == "__main__":
    # 加载特征文件
    feature_list = get_feature_list()

    # 加载模型文件
    kmeans = joblib.load('./data/kmeans')

    # 输出需要过滤掉的特征
    for f in feature_list:
        # 解析特征文件
        infos = f.split('|')
        filename = infos[0]
        feature = json.loads(infos[2])

        # 过滤掉不大正常的
        c = kmeans.predict([feature])
        if c== 4:
            RESULT_FILE.write(filename + infos[1] + "\n")
