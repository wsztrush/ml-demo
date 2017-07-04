import os
import numpy as np
from obspy import read
from matplotlib import pyplot as plt
import datetime

# /Users/tianchi.gzt/Downloads/preliminary/preliminary/before/XX.YZP.2008105000000.BHZ
# /Users/tianchi.gzt/Downloads/preliminary/preliminary/after/XX.YZP.2008105000000.BHZ
# print(float(datetime.datetime.fromtimestamp(1217404801.32 + 8 * 3600 + 673.21).strftime('%Y%m%d%H%M%S.%f')))

result = open("../data/result_1.csv", "a")

# 时间格式化
def format_time(t):
    return str(float(datetime.datetime.fromtimestamp(t + 8 * 3600).strftime('%Y%m%d%H%M%S.%f')))


# 处理单个文件
def process_day_file(dir, file):
    print(file)
    files = [dir + file + ".BHE", dir + file + ".BHN", dir + file + ".BHZ"]

    if not (os.path.exists(files[0]) and os.path.exists(files[1]) and os.path.exists(files[2])):
        return

    contents = [read(x) for x in files]
    datas = [np.abs(i[0].data - np.mean(i[0].data)) for i in contents]

    # 统计用来计算的数据
    starttime = float(contents[0][0].stats.starttime.strftime("%s.%f"))
    delta = contents[0][0].stats.delta
    dh = np.sqrt(np.power(datas[0], 2) + np.power(datas[1], 2))
    dv = datas[2]
    dh_mean = np.mean(dh)
    dv_mean = np.mean(dv)

    print(dh_mean, dv_mean)

    # TEST
    index = 0
    step = 100
    interval = 5
    x = np.arange(step)
    while index < len(dv):
        m1 = np.mean(dv[index:index + interval])
        m2 = np.mean(dv[index:index + interval * 10])
        if m1 < dv_mean * 5 or m2 < dv_mean * 5:
            index += interval
            continue

        # plt.plot(x, [np.mean(dh[index + i * interval:index + (i + 1) * interval]) for i in np.arange(step)], 'b')
        # plt.plot(x, [np.mean(dv[index + i * interval:index + (i + 1) * interval]) for i in np.arange(step)], 'g')
        # plt.show()
        # print(index)
        result.write(file[3:6] + "," + format_time(starttime + index * delta) + ",P\n")
        result.flush()

        index += step * interval

        while index < len(dv):
            if np.mean(dv[index:index + interval]) < dv_mean * 3:
                break
            index += interval


# 处理目录
def process_dir(dir):
    files = os.listdir(dir)
    file_set = set()

    for i in files:
        file_set.add(i[:-4])

    for i in file_set:
        process_day_file(dir, i)


if __name__ == '__main__':
    process_dir("/Users/tianchi.gzt/Downloads/preliminary/preliminary/before/")
