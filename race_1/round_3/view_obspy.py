from obspy import read
import os


def process_day_file(dir, file):
    files = [dir + file + ".BHE", dir + file + ".BHN", dir + file + ".BHZ"]

    if not (os.path.exists(files[0]) and os.path.exists(files[1]) and os.path.exists(files[2])):
        return

    channel = read(files[2])
    channel.plot(type='dayplot')
    #
    # channel = read(files[1])
    # channel.plot(type='dayplot')
    #
    # channel = read(files[2])
    # channel.plot(type='dayplot')



# 处理目录
def process_dir(dir):
    process_day_file(dir, 'XX.YGD.2008187000000')


if __name__ == '__main__':
    process_dir("/Users/tianchi.gzt/Downloads/race_1/after/")
