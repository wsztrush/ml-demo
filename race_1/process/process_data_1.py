# 1. 读取文件
# 2. 读取相应特征指定范围内数据path,time,index,avg,delta,[-5000:5000]
import obspy
import os
import numpy
import time

BASE_PATH = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/"
RESULT_PATH = "./data/process_data_1.csv"

result = []

parse_time = lambda x: time.strptime() + input()


# 处理单个文件
def process_file(file_path):
    content = obspy.read(file_path)[0]

    data = numpy.abs(content.data)

    avg = data.mean()
    start_time = int(float(content.stats.starttime) * 1000)
    delta = int(float(content.stats.delta) * 1000)

    print(content.stats)


# 处理单个目录
def process_dir(dir_path):
    files = os.listdir(dir_path)[:3]

    for file_path in files:
        process_file(dir_path + file_path)


if __name__ == "__main__":
    process_dir(BASE_PATH + "before/")
    process_dir(BASE_PATH + "after/")
