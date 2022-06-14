import datetime
import math
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plot
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor

# Extract the required data

data = pd.read_csv("../data/dublinbikes_data.csv", usecols=[1, 3, 6], parse_dates=[1])
# UPPER SHERRARD STREET | HANOVER QUAY
df = data[data.NAME == str('UPPER SHERRARD STREET')].drop(columns=["NAME"])
print(df.columns)
# Get time difference between records
date_col = pd.DatetimeIndex(df.TIME).view(int) / 10 ** 9
time_diff = date_col[1] - date_col[0]

print(time_diff)

# Extract data between 28th January and March 14
df = df[((pd.DatetimeIndex(df.TIME) > datetime.datetime(2020, 1, 28)) &
         (pd.DatetimeIndex(df.TIME) < datetime.datetime(2020, 3, 14)))]

df = df[((pd.DatetimeIndex(df.TIME).dayofweek == 5) | (pd.DatetimeIndex(df.TIME).dayofweek == 6))]
# Extract the required columns
y = df["AVAILABLE BIKES"]
# y = df["TIME"]
print(np.shape(y))

q = 10
lag_weekly = 3
# Find length of feature vector
weekly_len = math.floor(2 * 24 * 60 * 60 / time_diff)
print("Weekly len is:", weekly_len)
q_in_steps = int(q * 60 / time_diff)
print(q_in_steps)
length = len(y) - lag_weekly * weekly_len - q_in_steps

X = y[q_in_steps: int(q_in_steps + length)]

for i in range(1, lag_weekly):
    temp = y[q_in_steps + i * weekly_len: int(q_in_steps + i * weekly_len + length)]
    X = np.column_stack((X, temp))

lag = 3
for i in range(lag - 1, -1, -1):
    print(i)
    temp = y[lag_weekly * weekly_len - i*3: int(lag_weekly * weekly_len - i*3 + length)]
    X = np.column_stack((X, temp))

yy = y[q_in_steps + lag_weekly * weekly_len: int(q_in_steps + lag_weekly * weekly_len + length)].to_numpy()
np.set_printoptions(threshold=sys.maxsize)

XX = X
temp_date_col = pd.DatetimeIndex(df.TIME).view(int) / 10 ** 9
l = len(y) - lag_weekly * 7 - q_in_steps

date_range_in_days = ((temp_date_col[
                      q_in_steps + lag_weekly * weekly_len : int(q_in_steps + lag_weekly * weekly_len + length) ] -
                      temp_date_col[q_in_steps + lag_weekly * weekly_len]) / (24 * 60 * 60)) + 4


mean_error = []
std_error = []
k_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for k in k_array:
    temp_error = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(XX):
        model = KNeighborsRegressor(n_neighbors=k).fit(XX[train], yy[train])
        y_pred = model.predict(XX[test])
        temp_error.append(mean_squared_error(y_true=yy[test], y_pred=y_pred))
    mean_error.append(np.mean(temp_error))
    std_error.append(np.std(temp_error))
print(mean_error)
plot.errorbar(k_array, mean_error, std_error)
plot.xlabel("Number of Neighbours(k)")
plot.ylabel("Mean Square Error")
plot.show()

gamma_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

mean_error_gamma = []
std_error_gamma = []
for gamma in gamma_array:
    temp_error = []
    kf = KFold(n_splits=5)


    def kernel(d):
        w = np.exp(-d ** 2 / gamma)
        return w


    for train, test in kf.split(XX):
        model = KNeighborsRegressor(n_neighbors=4, weights=kernel).fit(XX[train], yy[train])
        y_pred = model.predict(XX[test])
        temp_error.append(mean_squared_error(y_true=yy[test], y_pred=y_pred))
    mean_error_gamma.append(np.mean(temp_error))
    std_error_gamma.append(np.std(temp_error))

print(mean_error_gamma)
plot.errorbar(gamma_array, mean_error_gamma, std_error_gamma)
plot.xlabel("Gamma")
plot.ylabel("Mean Square Error")
plot.show()


def kernel(d):
    gamma = 5
    w = np.exp(-d ** 2 / gamma)
    return w


train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
model = KNeighborsRegressor(n_neighbors=4, weights=kernel).fit(XX[train], yy[train])
# model = KNeighborsRegressor(n_neighbors=5, weights='uniform').fit(XX[train], yy[train])
y_pred = model.predict(XX[test])
print(r2_score(y_true=yy[test], y_pred=y_pred))
print(mean_squared_error(y_true=yy[test], y_pred=y_pred))
print(mean_squared_error(y_true=yy[test], y_pred=y_pred, squared=False))

print(len(date_range_in_days))
print(len(yy))
print(l)
y_pred_full = model.predict(XX)
plot.scatter(date_range_in_days, yy, color="blue")
plot.scatter(date_range_in_days, y_pred_full, color="red")
plot.show()
