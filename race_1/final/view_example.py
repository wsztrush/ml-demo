import os
import numpy as np
from obspy import read
from matplotlib import pyplot as plt
import tensorflow as tf

dir = "/Users/tianchi.gzt/Downloads/example30/"
files = os.listdir(dir)

for file in files:
    content = read(dir + file)
    print(content[0].stats)

    b = content[0].meta.sac['b']
    s = (content[0].meta.sac['t0'] - b) * 100
    p = (content[0].meta.sac['a'] - b) * 100

    y = content[0].data
    l = len(y)
    x = np.arange(0, l)

    plt.plot(x, y, '#0000FF')

    interval = 5
    step = int(l / interval)

    x1 = np.arange(5, l, interval)
    print(len(x1))
    y1 = np.zeros(l)

    for i in np.arange(l):
        y1[i] = np.std(y[i:i + interval])  # np.max(y[i:i + interval]) - np.min(y[i:i + interval])

    plt.plot(x, y1, '#FF00FF')

    if p > 0:
        plt.plot(p, 0, 'ro')
    if s > 0:
        plt.plot(s, 0, 'yo')
    plt.show()
