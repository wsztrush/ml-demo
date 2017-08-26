import numpy as np
import race_util
import os

from sklearn.externals import joblib
from sklearn.neural_network import BernoulliRBM
from matplotlib import pyplot as plt


def get_feature(stock_value, left, right):
    left = int(left - (right - left) / 19)
    if left >= 0:
        right -= (right - left) % 20

        tmp = np.mean(stock_value[left:right].reshape(20, -1), axis=1)

        return (tmp / (np.max(tmp) + 1.0)).tolist()


def get_x():
    result = []

    for unit in os.listdir(race_util.range_path):
        stock_value = np.load(race_util.stock_path + unit)[0]
        range_value = np.load(race_util.range_path + unit)

        for left, right in range_value:
            f = get_feature(stock_value, left, right)
            if f:
                result.append(f)

    return result


def main():
    x = get_x()

    print(len(x))

    model = BernoulliRBM(n_components=40, learning_rate=0.01, batch_size=1000, n_iter=200, verbose=1, random_state=None)
    model.fit(x)

    print(model)

    joblib.dump(model, race_util.rbm_file)


def check():
    x = get_x()
    model = joblib.load(race_util.rbm_file)

    for i in x:
        t = model.transform([i])[0]

        plt.subplot(2, 1, 1)
        plt.bar(np.arange(len(i)), i)

        plt.subplot(2, 1, 2)
        plt.bar(np.arange(len(t)), t)
        plt.show()


if __name__ == '__main__':
    race_util.config()
    main()
    # check()
