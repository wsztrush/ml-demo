# 根据生成的Filter对结果进行过滤，用来生成最终的结果文件
#
# 1. 读取所有的过滤文件
# 2. 读取特征文件
# 3. 过滤feature文件

import json

FILTER_RESULT_FILE = open('./data/filter_result.txt', 'a')


def read_filter_file(filename):
    ret = set()
    filter_file = open('./data/' + filename, 'r')
    while True:
        line = filter_file.readline()
        if not line:
            break

        ret.add(line.strip())

    return ret


def filter_feature(filter_set):
    feature_file = open('./data/feature.txt', 'r')
    while True:
        line = feature_file.readline()
        if not line:
            break

        infos = line.split('|')
        key = infos[0] + infos[1]

        if key in filter_set:
            continue

        FILTER_RESULT_FILE.write(line)


if __name__ == '__main__':
    # 读取过滤文件
    filter_set = read_filter_file('filter_1.txt')

    # 过滤特征文件
    filter_feature(filter_set)
