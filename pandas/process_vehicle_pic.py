import pandas as pd
import numpy as np
import math
import cv2

from skimage import io, transform

FILE_PATH = '/Users/tianchi.gzt/Downloads/2017-09-28-11-01-49_EXPORT_CSV_1032996_473_0.csv'
OSS_PATH = 'https://ipms.oss-cn-shanghai.aliyuncs.com/'

if __name__ == '__main__':
    a = pd.read_csv(FILE_PATH)
    a = a['driving_license_pic']
    b = set()

    for i in a:
        if i is None or (isinstance(i, float) and math.isnan(i)):
            continue
        if i.startswith('http'):
            b.add(i)
        else:
            b.add(OSS_PATH + i)

    print(len(b))

    for i in b:


        img = io.imread(i, as_grey=True)
        io.imshow(img)
        io.show()
