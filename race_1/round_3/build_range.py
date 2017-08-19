import numpy as np
import os
import time
import multiprocessing

STD_PATH = "./data/std/"


def process(unit):
    start = time.time()

    file_stds = np.fromfile(STD_PATH + unit)

    print(unit, time.time() - start)


def main():
    unit_set = set()
    for filename in os.listdir(STD_PATH):
        unit_set.add(filename)

    pool = multiprocessing.Pool(processes=4)
    pool.map(process, unit_set)


if __name__ == '__main__':
    main()
