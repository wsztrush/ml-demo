import pandas as pd
import numpy as np
import math
from PIL import Image
import pytesseract

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
        img = io.imread(i)  # , as_grey=True
        ret = pytesseract.image_to_string(Image.fromarray(img), lang='chi_sim')

        # io.imshow(img)
        # io.show()

        print(i)

        if len(ret) >= 0 and ('中' in ret or '华' in ret or '人' in ret or '民' in ret or '共' in ret or '和' in ret or '国' in ret or '机' in ret or '动' in ret or '车' in ret or '行' in ret or '驶' in ret or '证' in ret):
            io.imshow(img)
            io.show()
