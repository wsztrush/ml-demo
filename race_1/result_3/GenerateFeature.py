# 生成特征文件
#
# 1. 读取范围文件
# 2. 读取原始数据文件
# 3. 生成特征，并写入文件，格式为：文件名|范围|特征一|特征二|特征三

from obspy import read
import numpy as np
import json
import multiprocessing
import os

DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"
FEATURE_PATH = "./data/feature.txt"

INTERVAL = 5
L = multiprocessing.Lock()
if os.path.exists(FEATURE_PATH):
    os.remove(FEATURE_PATH)
RESULT_FILE = open(FEATURE_PATH, "w")


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


# 根据文件及范围信息生成对应的特征文件。
def process(line):
    kv = line.split('|')
    filename = kv[0]
    range_list = json.loads(kv[1])

    if len(range_list) == 0:
        return

    file_content = read(DIR_PATH + filename)
    data = file_content[0].data

    result = []
    for r in range_list:
        left, right = r[0], r[1]

        left = max(0, int(left - (right - left) * 0.2))
        left = left - left % INTERVAL

        print(filename, left, right)

        tmp = data[left:right].reshape(-1, INTERVAL)
        tmp_mean = [np.std(tmp, axis=1)][0]
        tmp_mean_len = len(tmp_mean) - len(tmp_mean) % 10
        tmp_mean = tmp_mean[:tmp_mean_len].reshape(10, -1)

        # 根据之前的结果进行过滤
        key = filename + json.dumps((left, right))
        if key in filter_set:
            continue

        # 特征一：分成10等分，分别计算与最大值的比例
        _10_mean_height = [np.mean(tmp_mean, axis=1)][0]
        _10_mean_rate_feature = _10_mean_height / np.max(_10_mean_height)

        # 特征二：平均振幅
        _10_mean_real_feature = np.log10(_10_mean_height + 1)

        # 特征三：分别计算10等分的振幅
        # _10_std_real = data[left:right]
        # _10_std_real = _10_std_real[:len(_10_std_real) - len(_10_std_real) % 10]
        # _10_std_real = [np.std(_10_std_real.reshape(10, -1), axis=1)][0]
        # _10_std_real_feature = np.log10(_10_std_real + 1)

        result.append(filename + "|" +
                      json.dumps((left, right)) + "|" +
                      json.dumps(_10_mean_rate_feature.tolist()) + "|" +
                      json.dumps(_10_mean_real_feature.tolist())
                      )

    # 将特征写出到文件
    L.acquire()
    for r in result:
        RESULT_FILE.write(r + "\n")
    RESULT_FILE.flush()
    L.release()


if __name__ == '__main__':

    range_file = open("./data/range.txt", "r")
    range_file_content = []

    filter_set = get_all_filter()

    while 1:
        line = range_file.readline()
        if not line:
            break
        range_file_content.append(line)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, range_file_content)
