from obspy import read
import numpy as np
import os
import time
import multiprocessing

DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"


def process(unit):
    start = time.time()

    file_names = [DIR_PATH + unit + ".BHE", DIR_PATH + unit + ".BHN", DIR_PATH + unit + ".BHZ"]
    file_contents = [read(i) for i in file_names]
    file_datas = [i[0].data for i in file_contents]
    file_stds = np.array([[np.std(data.reshape(-1, 5), axis=1)][0] for data in file_datas])

    np.save("./data/std/" + unit, file_stds)

    print(unit, time.time() - start)


def main():
    unit_set = set()
    for filename in os.listdir(DIR_PATH):
        unit = filename[:-4]
        unit_set.add(unit)

    print(len(unit_set))

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)


if __name__ == '__main__':
    main()
