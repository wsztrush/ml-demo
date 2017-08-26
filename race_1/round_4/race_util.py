import matplotlib

origin_dir_path = "/Users/tianchi.gzt/Downloads/race_1/after/"
stock_path = "./data/shock/"
range_path = "./data/range/"

rbm_file = './data/rbm'
sample_1_file = './data/sample_1.npy'

model_1 = './data/model_1'

stock_step = 10
stock_mean_step = 1
stock_mean_gap = 2


def config():
    p = matplotlib.rcParams
    p["figure.figsize"] = (15, 8)


def get_before_left(left, right):
    return max(int(left - (right - left) * 0.1), 0)
