import numpy as np
import os
import time
import multiprocessing
import race_util
import build_model_1

from sklearn.externals import joblib

model_1 = joblib.load('./data/model_1')


def get_result_range(unit):
    if os.path.exists('./data/result_range/' + unit):
        return np.load('./data/result_range/' + unit).tolist()
    else:
        return []


def process(unit):
    start_time = time.time()

    result = get_result_range(unit)
    all_range_list = np.load('./data/all_range/' + unit)
    shock_value = np.load('./data/shock/' + unit)

    for left, right in all_range_list:
        if [left, right] in result:
            continue
        else:
            feature = [build_model_1.build_feature(shock_value, left, right)]
            predict_ret = model_1.predict(feature)[0]

            if predict_ret == 0:
                result.append([left, right])

    if len(result) > 0:
        np.save('./data/result_range/' + unit, result)

    print('[COST]', unit, len(result), time.time() - start_time)


def main():
    # pool = multiprocessing.Pool(processes=4)
    # pool.map(process, os.listdir('./data/all_range/'))
    process('GS.WDT.2008183000000.npy')


if __name__ == '__main__':
    main()
