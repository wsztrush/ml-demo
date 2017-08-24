import matplotlib

STD_PATH = "./data/std/"
RANGE_PATH = "./data/range/"
PRE_RANGE_PATH = "./data/pre_range/"

DIR_PATH = "/Users/tianchi.gzt/Downloads/race_1/after/"

GBDT_MODEL_FILE = "./data/gbdt"
MODEL_SAMPLE_FILE = "./data/clf_sample.npy"


def get_left_right(l, r):
    return max(int(l - (r - l) * 0.1), 0), r


def config():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 8)


def gene_feature_1(file_std, left, right):
    pass
