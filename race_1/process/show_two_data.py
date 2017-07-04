from obspy import read
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

pause = False


# 加载数据
def load_data():
    content_e = read("/Users/tianchi.gzt/Downloads/example30/30.QCH.BHE.SAC")
    content_e = np.abs(content_e[0].data - np.mean(content_e[0].data))
    content_n = read("/Users/tianchi.gzt/Downloads/example30/30.QCH.BHN.SAC")
    content_n = np.abs(content_n[0].data - np.mean(content_n[0].data))
    content_h = np.sqrt(np.power(content_e, 2) + np.power(content_n, 2))

    content_v = read("/Users/tianchi.gzt/Downloads/example30/30.QCH.BHZ.SAC")
    content_v = np.abs(content_v[0].data - np.mean(content_v[0].data))

    return content_h, content_v


data = load_data()
step = 500


# 循环读数据
def next_value():
    index = 0
    interval = 10
    while index < len(data[0]):
        # 返回一定范围内的数据
        h_mean = np.zeros(step)
        v_mean = np.zeros(step)

        for i in np.arange(0, step):
            h_mean[i] = np.mean(data[0][index + i * interval:index + (i + 1) * interval])
            v_mean[i] = np.mean(data[1][index + i * interval:index + (i + 1) * interval])
            # z_mean[i] = np.mean(data[2][index + i * interval:index + (i + 1) * interval])
        yield h_mean, v_mean

        index += step * interval


x = np.arange(step)

fig = plt.figure()
ax = fig.add_subplot(111)
h_mean_line, = ax.plot([], [], 'b', label="h")
v_mean_line, = ax.plot([], [], 'g', label="v")
ax.set_ylim(0, 5000)
ax.set_xlim(0, step)


def refresh(next_value):
    h_mean, v_mean = next_value[0], next_value[1]

    h_mean_line.set_data(x, h_mean)
    v_mean_line.set_data(x, v_mean)

    return h_mean_line, v_mean_line


ani = animation.FuncAnimation(fig, refresh, next_value, blit=False, interval=500, repeat=False)
plt.show()
