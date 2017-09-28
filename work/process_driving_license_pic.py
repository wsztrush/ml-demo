import pandas as pd
import numpy as np
import math
from PIL import Image
import pytesseract

from skimage import io, transform


def get_urls():
    FILE_PATH = '/Users/tianchi.gzt/Downloads/2017-09-28-11-01-49_EXPORT_CSV_1032996_473_0.csv'
    OSS_PATH = 'https://ipms.oss-cn-shanghai.aliyuncs.com/'

    a = pd.read_csv(FILE_PATH)
    a = a['driving_license_pic']
    ret = set()

    for i in a:
        if i is None or (isinstance(i, float) and math.isnan(i)):
            continue
        if i.startswith('http'):
            ret.add(i)
        else:
            ret.add(OSS_PATH + i)

    return ret


def main():
    urls = get_urls()

    ret = []

    for i in urls:
        img = io.imread(i)
        image = Image.fromarray(img)

        a = pytesseract.image_to_string(image, lang='chi_sim')
        b = pytesseract.image_to_string(image, lang='eng')

        print(i)

        if len(a) > 0 and len(b) > 0:
            ret.append(i)

    print(len(ret))


if __name__ == '__main__':
    main()
