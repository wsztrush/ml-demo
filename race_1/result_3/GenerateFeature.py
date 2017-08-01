# 生成特征文件
#
# 1. 读取范围文件
# 2. 读取原始数据文件
# 3. 生成特征，并写入文件，格式为：文件名|范围|特征一|特征二|特征三

from matplotlib import pyplot as plt
from obspy import read
import numpy as np
import json

DIR_PATH = "/Users/tianchi.gzt/Downloads/preliminary/preliminary/after/"
INTERVAL = 5

