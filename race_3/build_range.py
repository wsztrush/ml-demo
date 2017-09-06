import numpy as np
import os
import time
import multiprocessing
import race_util
import build_model_1
import build_model_2

from sklearn.externals import joblib

model_1 = joblib.load('./data/model_1')
model_2 = [joblib.load('./data/model_2_' + str(i)) for i in np.arange(10)]


def process(unit):
    start_time = time.time()

    shock_value = np.load('./data/shock/' + unit)
    range_value = np.load('./data/all_range/' + unit)

    result = []
    for left, right in range_value:
        feature_1 = build_model_1.build_feature(shock_value, left, right)
        feature_2 = build_model_2.build_feature(shock_value, left, right)

        if feature_1 is None or feature_2 is None:
            continue

        predict_1 = model_1.predict([feature_1])[0]
        predict_2 = model_2[predict_1].predict([feature_2])[0]

        if build_model_2.flag[predict_1][predict_2] == 2:
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
