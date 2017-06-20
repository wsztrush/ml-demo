# coding: utf-8
# 昵称：成交量遥遥领先，单位：瓜子二手车(guazi.com)，在第二问中最终排名第五。
# 声明：代码仅作code review使用，注释用中文书写。如果需要提供正式的文档，请通知，我将用英文完成并提交pdf版本。

# 所需要的文件：
# volume(table 6)_training.csv
# volume(table 6)_training2.csv
# volume(table 6)_test2.csv
# 以上三个文件均为官方数据，放在同目录下即可。不需要使用天气数据，道路数据，以及第一问的数据。

# 运行代码所需要的版本：
# python version : Python 2.7.13
# scikit-learn version : scikit-learn (0.18.1)

# 代码文件： volume.py，仅一个python文件
# 运行方式： python volume.py
# 运行时间： 大约20秒，本地的机器为MacBook Pro（13英寸，2015年初期），处理器2.7GHz Intel Core i5，内存8GB 1867MHz DDR3
# 输出文件： volume.csv
# 这个输出文件将和我在5月31日下午4点测评的那次提交完全一致（注意：我的最好成绩是第二阶段中的第三次测评时的提交，并不是最后一次提交，这次提交的成绩为0.1326）
# 如果使用Python3或其他版本，也可能成功运行，但是输出的数小数点后所保留的位数可能不同（仅保留位数不同，数值本身是一样的），恳请主办方留意。

# 整体思路：将6:40~8:00的通过量作为feature（仅使用1个特征），8:00~8:20等6个时间段的通过量作为label，
# 下午的对应为15:40~17:00的通过量作为feature，17:00~17:20等6个时间段的通过量作为label，使用线性回归模型。
# 细节：将使用ETC和不使用ETC的数量分开，在部分道路上，将周末和工作日的数据分开，分别统计，训练，以及预测。
# 剔除节假日（国庆7天）的数据，剔除少量的噪音数据，按天剔除。
# 观察数据发现，在9月21日，9月28日的数据中，2,0收费站不使用ETC的车辆通过记录很可能被错误的统计在了1,0收费站中。
# 观察测试集发现，10月30日和10月31日极为可能也发生了同样的统计错误，本次预测做了这个假设。


from datetime import datetime, timedelta
from collections import defaultdict
import math
from sklearn import linear_model


def read_volume_datas(file_name):
    volume_datas = []
    f = open(file_name, "r")
    f.readline()
    for line in f:
        items = line.strip().replace('"', '').split(',')
        time = datetime.strptime(items[0], "%Y-%m-%d %H:%M:%S")
        tollgate_id = int(items[1])
        direction = int(items[2])
        vehicle_model = int(items[3])
        has_etc = int(items[4])
        vehicle_type = items[5]
        volume_datas.append((time, tollgate_id, direction, vehicle_model, has_etc, vehicle_type))
    f.close()
    return volume_datas


# 读取数据
volume_training = read_volume_datas("volume(table 6)_training.csv")
volume_validation = read_volume_datas("volume(table 6)_training2.csv")
volume_test = read_volume_datas("volume(table 6)_test2.csv")


# 获得对应的时间窗口开始的时间点
def get_start_time_window(time):
    time_window_minute = int(math.floor(time.minute / 20) * 20)
    start_time_window = datetime(time.year, time.month, time.day, time.hour, time_window_minute, 0)
    return start_time_window


# 数据类
class Data(object):
    def __init__(self):
        self._data = defaultdict(dict)
        self._rows = set()
        self._cols = set()

    def insert(self, row, col, value):
        self._data[row][col] = value
        self._rows.add(row)
        self._cols.add(col)

    def get_data(self, valid_rows=None, default=None):
        rows = sorted(self._rows)
        cols = sorted(self._cols)
        X = []
        for row in rows:
            if valid_rows and row not in valid_rows:
                continue
            x = []
            for col in cols:
                value = self._data[row].get(col, default)
                x.append(value)
            X.append(x)
        return X

    def get_rows(self):
        return self._rows

    def get_cols(self):
        return self._cols


# 对日期进行分类，1为节假日，2为周末，3为工作日
def get_date_type(date):
    if "2016-10-01" <= date and date <= "2016-10-07":
        return 1
    elif date in ["2016-09-24", "2016-09-25", "2016-10-15", "2016-10-16",
                  "2016-10-22", "2016-10-23", "2016-10-29", "2016-10-30"]:
        return 2
    else:
        return 3


# 对每一天的数据进行分类，同时去掉噪音数据。1不正常数据，2正常，3周末，4非周末。
# 按不同的收费站，方向，是否ETC，将数据分开，对每一部分的数据使用一个线性回归模型。
def get_volume_type(tollgate_id, direction, date, has_etc):
    if tollgate_id == 4 and direction == 0:  # tollgate_id == 4 为虚拟的道路，统计1,0和2,0道路通过量的总和。
        if date == "2016-09-30":  # 去除噪音数据
            return 1
        elif get_date_type(date) == 1:
            return 1
        else:
            return 2

    if tollgate_id == 1 and direction == 0 and has_etc == 0:
        if date in ["2016-09-21", "2016-09-27", "2016-09-28", "2016-09-30", "2016-10-30", "2016-10-31"]:  # 去除噪音数据
            return 1
        elif get_date_type(date) == 1:
            return 1
        else:
            return 2
    if tollgate_id == 1 and direction == 0 and has_etc == 1:
        if date in ["2016-09-23", "2016-09-25", "2016-09-27", "2016-09-30"]:  # 去除噪音数据
            return 1
        elif get_date_type(date) == 1:
            return 1
        else:
            return 2

    if tollgate_id == 2 and direction == 0 and has_etc == 0:
        if date in ["2016-09-21", "2016-09-24", "2016-09-27", "2016-09-28", "2016-09-30", "2016-10-15",
                    "2016-10-23", "2016-10-30", "2016-10-31"]:  # 去除噪音数据
            return 1
        elif get_date_type(date) == 1:
            return 1
        else:
            return 2
    if tollgate_id == 2 and direction == 0 and has_etc == 1:
        if date in ["2016-09-24", "2016-09-29", "2016-09-30", "2016-10-15"]:  # 去除噪音数据
            return 1
        elif get_date_type(date) == 1:
            return 1
        elif get_date_type(date) == 2:
            return 3
        else:
            return 4

    if tollgate_id == 3 and direction == 0 and has_etc == 0:
        if date in ["2016-09-29", "2016-09-30"]:  # 去除噪音数据
            return 1
        elif get_date_type(date) == 1:
            return 1
        elif get_date_type(date) == 2:
            return 3
        else:
            return 4

    if tollgate_id == 3 and direction == 0 and has_etc == 1:
        if date in ["2016-09-30"]:  # 去除噪音数据
            return 1
        elif get_date_type(date) == 1:
            return 1
        elif get_date_type(date) == 2:
            return 3
        else:
            return 4

    if tollgate_id == 1 and direction == 1 and has_etc == 0:
        if date in ["2016-09-20", "2016-09-30"]:  # 去除噪音数据
            return 1
        elif get_date_type(date) == 1:
            return 1
        else:
            return 2
    if tollgate_id == 1 and direction == 1 and has_etc == 1:
        if get_date_type(date) == 1:
            return 1
        elif get_date_type(date) == 2:
            return 3
        else:
            return 4

    if tollgate_id == 3 and direction == 1 and has_etc == 0:
        if date in ["2016-09-20", "2016-09-29", "2016-09-30"]:  # 去除噪音数据
            return 1
        elif get_date_type(date) == 1:
            return 1
        else:
            return 2
    if tollgate_id == 3 and direction == 1 and has_etc == 1:
        if date in ["2016-09-30"]:  # 去除噪音数据
            return 1
        elif get_date_type(date) == 1:
            return 1
        elif get_date_type(date) == 2:
            return 3
        else:
            return 4


# 计算label，即统计每个时间窗口的通过量，用ETC的和非ETC的分开统计
def get_volume_label(Y_all, datas, key_hour, test_hours):
    key_list = defaultdict(lambda: defaultdict(list))
    for line in datas:
        time, tollgate_id, direction, vehicle_model, has_etc, vehicle_type = line
        start_time_window = get_start_time_window(time)
        if start_time_window.hour not in test_hours:
            continue
        model_key = (tollgate_id, direction, start_time_window.hour, start_time_window.minute)
        key_list[model_key][start_time_window].append(has_etc)
        if tollgate_id in [1, 2] and direction == 0:
            model_key = (4, 0, start_time_window.hour, start_time_window.minute)
            key_list[model_key][start_time_window].append(has_etc)
    for model_key in key_list:
        data = Data()
        for start_time_window in key_list[model_key]:
            date = start_time_window.strftime("%Y-%m-%d")
            values = key_list[model_key][start_time_window]
            sum_value = sum([1 for v in values if v == 0])
            data.insert(date, 0, sum_value)  # 不使用ETC的通过量
            sum_value = sum([1 for v in values if v == 1])
            data.insert(date, 1, sum_value)  # 使用ETC的通过量
        Y_all[model_key] = data
    return


# 获取特征，仅一个特征，上午是6:40~8:00的通过量，下午是15:40~17:00的通过量。用ETC的和非ETC的同样分开统计。
def get_volume_feature(Y_all, datas, key_hour):
    key_list = defaultdict(lambda: defaultdict(list))
    for line in datas:
        time, tollgate_id, direction, vehicle_model, has_etc, vehicle_type = line
        start_time_window = get_start_time_window(time)
        if not ((start_time_window.hour == key_hour - 1)
                or (start_time_window.hour == key_hour - 2 and start_time_window.minute >= 40)):
            continue
        date = start_time_window.strftime("%Y-%m-%d")
        for hour in range(key_hour, key_hour + 2, 1):
            for minute in [0, 20, 40]:
                model_key = (tollgate_id, direction, hour, minute)
                key_list[model_key][date].append(has_etc)
                if tollgate_id in [1, 2] and direction == 0:
                    model_key = (4, 0, hour, minute)
                    key_list[model_key][date].append(has_etc)
    for model_key in key_list:
        tollgate_id, direction, hour, minute = model_key
        data = Data()
        for date in key_list[model_key]:
            values = key_list[model_key][date]
            sum_value = sum([1 for v in values if v == 0])
            data.insert(date, 0, sum_value)  # 不使用ETC的通过量
            sum_value = sum([1 for v in values if v == 1])
            data.insert(date, 1, sum_value)  # 使用ETC的通过量
            volume_type = get_volume_type(tollgate_id, direction, date, 0)
            data.insert(date, 2, volume_type)  # 不使用ETC的通过量，当天的类别
            volume_type = get_volume_type(tollgate_id, direction, date, 1)
            data.insert(date, 3, volume_type)  # 使用ETC的通过量，当天的类别
        Y_all[model_key] = data
    return


# 产生训练样本
Y_volume_training = {}
Y_volume_validation = {}

X_volume_training = {}
X_volume_validation = {}
X_volume_test = {}

get_volume_label(Y_volume_training, volume_training, 8, [8, 9])  # 训练集上午时间段的label
get_volume_label(Y_volume_training, volume_training, 17, [17, 18])  # 训练集下午时间段的label
get_volume_label(Y_volume_validation, volume_validation, 8, [8, 9])  # 验证集上午时间段的label
get_volume_label(Y_volume_validation, volume_validation, 17, [17, 18])  # 验证集下午时间段的label

get_volume_feature(X_volume_training, volume_training, 8)  # 训练集上午时间段的feature
get_volume_feature(X_volume_training, volume_training, 17)  # 训练集下午时间段的feature
get_volume_feature(X_volume_validation, volume_validation, 8)  # 验证集上午时间段的feature
get_volume_feature(X_volume_validation, volume_validation, 17)  # 验证集下午时间段的feature

get_volume_feature(X_volume_test, volume_test, 8)  # 测试集上午时间段的feature
get_volume_feature(X_volume_test, volume_test, 17)  # 测试集下午时间段的feature


def combine_date(X_data, Y_data, default=None):
    X = {}
    Y = {}
    date = {}
    for model_key in X_data:
        x_rows = X_data[model_key].get_rows()
        y_rows = Y_data[model_key].get_rows()
        rows = x_rows & y_rows
        date[model_key] = sorted(rows)
        X[model_key] = X_data[model_key].get_data(rows, default)
        Y[model_key] = Y_data[model_key].get_data(rows, default)
    return X, Y, date


# 将同一天上午的数据，下午的数据（feature和label）分别合到一起，产生所有的训练样本。
X_training, Y_training, date_training = combine_date(X_volume_training, Y_volume_training, 0)
X_validation, Y_validation, date_validation = combine_date(X_volume_validation, Y_volume_validation, 0)
X_test = {model_key: X_volume_test[model_key].get_data(None, 0) for model_key in X_volume_test}
date_test = {model_key: sorted(X_volume_test[model_key].get_rows()) for model_key in X_volume_test}

# 最终的训练用上训练集和验证集的数据。
X_training_all = {model_key: X_training[model_key] + X_validation[model_key] for model_key in X_training}
Y_training_all = {model_key: Y_training[model_key] + Y_validation[model_key] for model_key in Y_training}


# 模型类
class Model(object):
    def __init__(self):
        self.total = 9
        self.regs = [0] * (self.total)

    def fit(self, X, Y):
        self.X = [[] for i in range(self.total)]
        self.Y = [[] for i in range(self.total)]
        for i, x in enumerate(X):
            w = x[2]
            self.X[w].append([X[i][0]])
            self.Y[w].append([Y[i][0]])
            w = x[3] + 4
            self.X[w].append([X[i][1]])
            self.Y[w].append([Y[i][1]])
        for i in range(self.total):
            if len(self.X[i]) > 0:
                # 对每一种类型的数据用一个线性回归模型进行训练。
                self.regs[i] = linear_model.LinearRegression(fit_intercept=True, normalize=False)
                self.regs[i].fit(self.X[i], self.Y[i])
        return

    def predict(self, X, model_key=None):
        predicted = []
        for i, x in enumerate(X):
            w = x[2]
            # 预测不使用ETC的通过量
            value0 = self.regs[w].predict([[x[0]]])[0][0]
            w = x[3] + 4
            # 预测使用ETC的通过量
            value1 = self.regs[w].predict([[x[1]]])[0][0]
            # 如果真实值偏小，预测的损失会更大一些，所以这里选择稍微预测小一点，乘以了0.96
            value0 *= 0.96
            value1 *= 0.96
            predicted.append([value0, value1])
        return predicted


# 对测试集进行训练，预测
Y_test = {}
for model_key in sorted(X_training):
    reg = Model()
    reg.fit(X_training_all[model_key], Y_training_all[model_key])
    predicted = reg.predict(X_test[model_key])
    Y_test[model_key] = predicted

# 对10月30日，31日的数据进行修复
for i in [8, 9, 17, 18]:
    for j in [0, 20, 40]:
        # 1,0 不使用ETC的通过量修正为 1,0和2,0 不使用ETC的通过量的总和
        Y_test[(1, 0, i, j)][5][0] = Y_test[(4, 0, i, j)][5][0]
        Y_test[(1, 0, i, j)][6][0] = Y_test[(4, 0, i, j)][6][0]
        # 2,0 不使用ETC的通过量修正为0
        Y_test[(2, 0, i, j)][5][0] = 0
        Y_test[(2, 0, i, j)][6][0] = 0


# 输出结果
def output_volume(X_test, Y_test):
    begin_day = datetime(2016, 10, 25)
    f = open("volume.csv", "w")
    f.write("tollgate_id,time_window,direction,volume\n")
    for model_key in sorted(Y_test):
        rows = sorted(X_test[model_key].get_rows())
        tollgate_id, direction, hour, minute = model_key
        if tollgate_id == 4:
            continue
        for i, _ in enumerate(Y_test[model_key]):
            day = rows[i]
            # 预测结果为，使用ETC和不使用ETC的通过量的总和
            value = sum(Y_test[model_key][i])
            start_time_window = begin_day + timedelta(hours=hour, minutes=minute)
            end_time_window = begin_day + timedelta(hours=hour, minutes=minute + 20)
            f.write("{0},\"[{5} {1},{5} {2})\",{3},{4}\n".format(
                tollgate_id,
                start_time_window.strftime("%H:%M:%S"),
                end_time_window.strftime("%H:%M:%S"),
                direction, value, day
            ))
    f.close()


output_volume(X_volume_test, Y_test)

# 结束
