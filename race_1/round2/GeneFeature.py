from obspy import read
import numpy as np
import json
import multiprocessing
import os

DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"
RESULT_PATH = "./data/feature.txt"
INTERVAL = 5
L = multiprocessing.Lock()
if os.path.exists("./data/feature.txt"):
    os.remove("./data/feature.txt")
RESULT_FILE = open("./data/feature.txt", "w")


# 获取所有可以过滤掉的区间
def get_all_filter():
    result = set()

    filename_list = os.listdir("./data/")

    if not filename_list:
        return result

    # 遍历所有'filter'开头的文件
    for filename in filename_list:
        if filename.startswith('filter'):
            filter_file = open('./data/' + filename)

            while True:
                filter_line = filter_file.readline()

                if not filter_line:
                    break

                filter_line = filter_line.strip()
                if filter_line in result:
                    continue

                result.add(filter_line)
            filter_file.close()

    return result


# 计算特征
def get_feature_1(file_stds, left, right):
    right -= (right - left) % 5
    result = [[np.mean(file_std[left:right].reshape(5, -1), axis=1)][0] for file_std in file_stds]
    result = [i / (np.max(i) + 1) for i in result]
    return result


# 处理单元
def process(line):
    # 解析文件内容。
    infos = line.split('|')
    unit = infos[0]
    range_list = json.loads(infos[1])

    if len(range_list) == 0:
        return

    # 读取数据并进行初步加工。
    file_names = [DIR_PATH + unit + ".BHE", DIR_PATH + unit + ".BHN", DIR_PATH + unit + ".BHZ"]
    file_conents = [read(i) for i in file_names]
    file_datas = [i[0].data for i in file_conents]
    file_stds = [[np.std(i.reshape(-1, INTERVAL), axis=1)][0] for i in file_datas]

    # 生成特征。
    result = []
    for r in range_list:
        # 计算
        left, right = r[0], r[1]

        left = max(0, int(left - (right - left) * 0.2))
        left = left - left % INTERVAL

        # 过滤
        key = unit + json.dumps((left, right))
        if key in filter_set:
            continue

        # 特征一
        feature_1 = get_feature_1(file_stds, int(left / INTERVAL), int(right / INTERVAL))
        result.append(
            unit + "|" +
            json.dumps((left, right)) + "|" +
            json.dumps(feature_1[0].tolist()) + "|" +
            json.dumps(feature_1[1].tolist()) + "|" +
            json.dumps(feature_1[2].tolist())
        )

    L.acquire()
    for r in result:
        RESULT_FILE.write(r + "\n")
    RESULT_FILE.flush()
    L.release()


def main():
    range_file = open("./data/range.txt", "r")
    range_file_content = []
    while 1:
        line = range_file.readline()
        if not line:
            break
        range_file_content.append(line)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, range_file_content)


if __name__ == '__main__':
    filter_set = get_all_filter()

    main()

    RESULT_FILE.close()
