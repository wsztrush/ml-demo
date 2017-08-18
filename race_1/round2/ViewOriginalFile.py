import os
import numpy as np
from obspy import read

if __name__ == '__main__':
    dir_path = '/Users/tianchi.gzt/Downloads/race_1/after/'
    file_name_list = os.listdir(dir_path)

    for i in file_name_list:
        file_content = read(dir_path + i)

        # if file_content[0].stats.delta != 0.01:
        # print(file_content[0].stats.starttime)

    # for i in np.arange(0, len(file_name_list), 3):
    #     print(i)
    # i = 0
    # while i < len(file_name_list):
    #     a = file_name_list[i]
    #     b = file_name_list[i + 1]
    #     c = file_name_list[i + 2]
    #
    #     address_time = a[:-3]
    #     if b[:-3] != address_time or c[:-3] != address_time:
    #         print(a, b, c)
    #
    #     i += 3
