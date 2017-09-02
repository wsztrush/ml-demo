from obspy import read
import numpy as np
import os
import time
import multiprocessing
import race_util


def process(unit):
    start_time = time.time()
    file_names = [race_util.origin_dir_path + unit + ".BHE", race_util.origin_dir_path + unit + ".BHN", race_util.origin_dir_path + unit + ".BHZ"]
    file_contents = [read(i) for i in file_names]
    file_datas = [i[0].data for i in file_contents]

    ret = []
    for file_data in file_datas:
        tmp = file_data.reshape(-1, race_util.shock_step)

        ret.append(np.max(tmp, axis=1) - np.min(tmp, axis=1))

    if len(ret) != 3 or len(ret[0]) != 864000 or len(ret[1]) != 864000 or len(ret[2]) != 864000:
        print('[BAD CASE]', unit)
        return

    ret = np.sqrt(np.square(ret[0]) + np.square(ret[1]) + np.square(ret[2]))
    np.save('./data/shock/' + unit, ret)

    print('[COST]', unit, time.time() - start_time)


def main():
    unit_set = set()
    for filename in os.listdir(race_util.origin_dir_path):
        unit = filename[:-4]
        unit_set.add(unit)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)


if __name__ == '__main__':
    main()
