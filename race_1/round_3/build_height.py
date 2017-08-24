import numpy as np
import os
import time
import multiprocessing
import race_util

from obspy import read


def process(unit):
    file_names = [race_util.DIR_PATH + unit + ".BHE", race_util.DIR_PATH + unit + ".BHN", race_util.DIR_PATH + unit + ".BHZ"]
    file_contents = [read(i) for i in file_names]
    file_datas = [i[0].data for i in file_contents]


def main():
    unit_set = set()
    for filename in os.listdir(race_util.DIR_PATH):
        unit = filename[:-4]
        unit_set.add(unit)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)


if __name__ == '__main__':
    main()
