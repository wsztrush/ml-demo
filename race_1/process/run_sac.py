from obspy import read
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

pause = False


def load_data():
    content_e = read("/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/XX.XJI.2008229000000.BHE")
    content_n = read("/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/XX.XJI.2008229000000.BHN")
    content_z = read("/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/XX.XJI.2008229000000.BHZ")

    return np.abs(content_e[0].data - np.mean(content_e[0].data))


data = load_data()
step = 500


def next_value():
    index = 0
    interval = 5
    while index < len(data):
        print(index)
        # 找到有一定震动幅度的位置
        while index < len(data):
            mean = np.mean(data[index:index + interval])
            if mean > 400:
                break
            index += interval

        # 返回一定范围内的数据
        mean = np.zeros(step)
        std = np.zeros(step)

        for i in np.arange(0, step):
            mean[i] = np.mean(data[index + i * interval:index + (i + 1) * interval])
            std[i] = np.std(data[index + i * interval:index + (i + 1) * interval])
        yield mean, std

        index += step * interval


x = np.arange(step)

fig = plt.figure()
ax = fig.add_subplot(111)
mean_line, = ax.plot([], [])
std_line, = ax.plot([], [])
ax.set_ylim(0, 1000)
ax.set_xlim(0, step)


def refresh(next_value):
    mean, std = next_value[0], next_value[1]
    mean_line.set_data(x, mean)
    std_line.set_data(x, std)
    return mean_line


ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=100, repeat=False)
plt.show()
