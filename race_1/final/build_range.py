import numpy as np
import os
import time
import multiprocessing
import race_util
import build_model_1
import build_model_2
import build_result_range

from sklearn.externals import joblib
from obspy import read
from matplotlib import pyplot as plt

model_1 = joblib.load('./data/model_1')
model_2 = joblib.load('./data/model_2')


def process(unit):
    start_time = time.time()

    shock_value = np.load('./data/shock/' + unit)
    all_range_value = np.load('./data/all_range/' + unit)
    result_range = build_result_range.get_result_range(unit)

    result = []
    for left, right in all_range_value:
        # 第一个模型的过滤
        if [left, right] in result_range:
            continue

        # 第二个模型的过滤
        feature_2 = build_model_2.build_feature(shock_value, left, right)
        predict_2 = model_2.predict([feature_2])
        if predict_2 in [2, 15, 17]:
            continue

        result.append([left, right])

    if len(result) > 0:
        np.save('./data/range/' + unit, result)

    print('[COST]', unit, len(result), time.time() - start_time)


def main():
    # 生成所有可能的区间
    pool = multiprocessing.Pool(processes=4)
    pool.map(process, os.listdir('./data/all_range/'))
    # process('GS.WDT.2008183000000.npy')

    # 统计区间个数
    total = 0
    for unit in os.listdir('./data/range/'):
        range_list = np.load('./data/range/' + unit)

        total += len(range_list)

    print('[TOTAL RANGE]', total)


if __name__ == '__main__':
    race_util.config()

    main()
