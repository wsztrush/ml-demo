import numpy as np
import os
import time
import multiprocessing
import race_util
import build_model_1
import build_model_2

from sklearn.externals import joblib

model_1 = joblib.load('./data/model_1')


def process(unit):
    start_time = time.time()

    shock_value = np.load('./data/shock/' + unit)
    range_value = np.load('./data/all_range/' + unit)

    result = []
    for left, right in range_value:
        feature = build_model_1.build_feature(shock_value, left, right)

        if feature is None:
            continue

        predict = model_1.predict([feature])[0]
        before_ratio = build_model_1.build_before_ratio(shock_value, left, right)

        if predict not in [5, 8] and before_ratio[0] < 0.113:
            result.append([left, right])

    if len(result) > 0:
        np.save('./data/range/' + unit, result)

    print('[COST]', unit, len(result), time.time() - start_time)


def main():
    pool = multiprocessing.Pool(processes=4)
    pool.map(process, os.listdir('./data/all_range/'))

    total = 0
    for unit in os.listdir('./data/range/'):
        range_list = np.load('./data/range/' + unit)

        total += len(range_list)

    print('[TOTAL RANGE]', total)


if __name__ == '__main__':
    race_util.config()

    main()
