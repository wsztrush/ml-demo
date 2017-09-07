from obspy import read
import numpy as np
import os
import time
import multiprocessing
import race_util


def read_content(file_name):
    file_content = read(file_name)
    file_data = file_content[0].data

    tmp = file_data.reshape(-1, race_util.step)
    return np.std(tmp, axis=1)


def process(unit):
    start_time = time.time()

    a = read_content(race_util.origin_dir_path + unit + ".BHE")
    b = read_content(race_util.origin_dir_path + unit + ".BHN")
    np.save('./data/shock/' + unit, np.sqrt(np.square(a) + np.square(b)))

    print('[COST]', unit, time.time() - start_time)


def main():
    unit_set = set()
    for filename in os.listdir(race_util.origin_dir_path):
        unit = filename[:-4]
        unit_set.add(unit)

    print('unit_set.size = ', len(unit_set))

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)


if __name__ == '__main__':
    main()
