from obspy import read
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

pause = False


# 加载数据
def load_data():
    content_e = read("/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/XX.XJI.2008229000000.BHE")
    content_n = read("/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/XX.XJI.2008229000000.BHN")
    content_z = read("/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/XX.XJI.2008229000000.BHZ")

    return np.abs(content_e[0].data - np.mean(content_e[0].data)), np.abs(content_n[0].data - np.mean(content_n[0].data)), np.abs(content_z[0].data - np.mean(content_z[0].data))


data = load_data()
step = 100


# 循环读数据
def next_value():
    index = 0
    interval = 1
    while index < len(data[0]):
        # 返回一定范围内的数据
        e_mean = np.zeros(step)
        n_mean = np.zeros(step)
        z_mean = np.zeros(step)

        for i in np.arange(0, step):
            e_mean[i] = np.mean(data[0][index + i * interval:index + (i + 1) * interval])
            n_mean[i] = np.mean(data[1][index + i * interval:index + (i + 1) * interval])
            # z_mean[i] = np.mean(data[2][index + i * interval:index + (i + 1) * interval])
        yield e_mean, n_mean, z_mean

        index += step * interval


x = np.arange(step)

fig = plt.figure()
ax = fig.add_subplot(111)
e_mean_line, = ax.plot([], [])
n_mean_line, = ax.plot([], [])
z_mean_line, = ax.plot([], [])
ax.set_ylim(0, 1000)
ax.set_xlim(0, step)


def refresh(next_value):
    e_mean, n_mean, z_mean = next_value[0], next_value[1], next_value[2]

    e_mean_line.set_data(x, e_mean)
    n_mean_line.set_data(x, n_mean)
    z_mean_line.set_data(x, z_mean)

    return e_mean_line, n_mean_line, z_mean_line


ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=500, repeat=False)
plt.show()
