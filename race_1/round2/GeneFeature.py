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
    feature_size = 20

    right -= (right - left) % feature_size

    tmp = [i[left:right] for i in file_stds]
    tmp = [np.square(i) for i in tmp]
    tmp = (tmp[0] + tmp[1] + tmp[2]) / 3
    # tmp = np.sqrt(tmp)
    # tmp_max = np.max(tmp)
    # tmp = [np.max(tmp.reshape(feature_size, -1), axis=1)][0]
    tmp = np.max(tmp)

    ret = np.array([np.log(tmp + 1)])

    if np.isnan(ret).any() or not np.isfinite(ret).all():
        print(ret)

    return ret


# 计算特征
# def get_feature_2(file_stds, left, right):
#     right -= (right - left) % 10
#
#     tmp = [i[left:right] for i in file_stds]
#
#     result = np.array([max(file_stds[0][i], file_stds[1][i], file_stds[2][i]) for i in np.arange(left, right)])
#     result = result.reshape(50, -1)
#     result = [np.mean(result, axis=1)][0]
#     return result / np.max(result)


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

        if right - left < 1000:
            continue

        left = max(0, int(left - (right - left) * 0.2))
        left = left - left % INTERVAL

        # 过滤
        key = unit + json.dumps((left, right))
        if key in filter_set:
            continue

        file_std_max = np.max([np.max(i[int(left / INTERVAL):int(right / INTERVAL)]) for i in file_stds])
        if file_std_max < 1000:
            continue

        # 特征一
        feature_1 = get_feature_1(file_stds, int(left / INTERVAL), int(right / INTERVAL))
        result.append(
            unit + "|" +
            json.dumps((left, right)) + "|" +
            json.dumps(feature_1.tolist())
        )

        # 特征二
        # feature_2 = get_feature_2(file_stds, int(left / INTERVAL), int(right / INTERVAL))
        # result.append(
        #     unit + "|" +
        #     json.dumps((left, right)) + "|" +
        #     json.dumps(feature_2.tolist())
        # )

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
