import numpy as np
import os
import time
import multiprocessing
import race_util
import build_model_a
import build_model_b

from sklearn.externals import joblib

model_a = joblib.load('./data/model_a')
model_b = joblib.load('./data/model_b')


def process(unit):
    start_time = time.time()

    shock_value = np.load('./data/shock/' + unit)
    range_value = np.load('./data/all_range/' + unit)

    result = []
    for left, right in range_value:
        if build_model_a.predict(model_a, shock_value, left, right):
            continue

        if build_model_b.predict(model_b, shock_value, left, right):
            continue

        result.append([left, right])

    if len(result) > 0:
        np.save('./data/range/' + unit, result)

    print('[COST]', unit, len(result), time.time() - start_time)


def main():
    # 生成所有可能的区间
    pool = multiprocessing.Pool(processes=4)
    pool.map(process, os.listdir('./data/all_range/'))

    # 统计区间个数
    total = 0
    for unit in os.listdir('./data/range/'):
        range_list = np.load('./data/range/' + unit)

        total += len(range_list)

    print('[TOTAL RANGE]', total)


if __name__ == '__main__':
    race_util.config()

    main()