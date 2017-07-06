import os
import numpy as np
from obspy import read
from matplotlib import pyplot as plt

content = read("/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/XX.JMG.2008231000000.BHE")
content.plot(type='dayplot')

data = content[0].data

interval = 5
step = 1000

x = np.arange(0, step * interval, interval)
xx = np.arange(step * interval)

for i in np.arange(0, len(data), step * interval):
    y = np.zeros(step)
    max_y = 0

    for j in np.arange(step):
        y[j] = np.std(data[i + j * interval:i + (j + 1) * interval])
        max_y = max(y[j], max_y)

    # print(max_y)

    if max_y > 500:
        print(i, max_y)
        # plt.plot(xx, data[i:i + step * interval])
        # plt.plot(x, y)
        # plt.show()
