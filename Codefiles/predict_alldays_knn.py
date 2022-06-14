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

# Extract data between 28th January and March 14
df = df[((pd.DatetimeIndex(df.TIME) > datetime.datetime(2020, 1, 28)) &
         (pd.DatetimeIndex(df.TIME) < datetime.datetime(2020, 3, 14)))]

# Predictions for the weekdays
df_wd = df[((pd.DatetimeIndex(df.TIME).dayofweek != 5) & (pd.DatetimeIndex(df.TIME).dayofweek != 6))]
# Extract the required columns
y_wd = df_wd["AVAILABLE BIKES"]
# y = df["TIME"]

q_wd = 60
lag_weekly_wd = 2
# Find length of feature vector
weekly_len_wd = math.floor(5 * 24 * 60 * 60 / time_diff)
q_in_steps_wd = int(q_wd * 60 / time_diff)
length_wd = len(y_wd) - lag_weekly_wd * weekly_len_wd - q_in_steps_wd

X_wd = y_wd[q_in_steps_wd: int(q_in_steps_wd + length_wd)]

for i in range(1, lag_weekly_wd):
    temp = y_wd[q_in_steps_wd + i * weekly_len_wd: int(q_in_steps_wd + i * weekly_len_wd + length_wd)]
    X_wd = np.column_stack((X_wd, temp))

lag_daily_wd = 1
daily_len_wd = math.floor(24 * 60 * 60 / time_diff)

for i in range(lag_daily_wd, 0, -1):
    temp = y_wd[q_in_steps_wd + lag_weekly_wd * weekly_len_wd - i * daily_len_wd: int(
        q_in_steps_wd + lag_weekly_wd * weekly_len_wd - i * daily_len_wd + length_wd)]
    X_wd = np.column_stack((X_wd, temp))

lag_wd = 1
for i in range(lag_wd - 1, -1, -1):
    temp = y_wd[lag_weekly_wd * weekly_len_wd - i * 3: int(lag_weekly_wd * weekly_len_wd - i * 3 + length_wd)]
    X_wd = np.column_stack((X_wd, temp))

yy_wd = y_wd[q_in_steps_wd + lag_weekly_wd * weekly_len_wd:
             int(q_in_steps_wd + lag_weekly_wd * weekly_len_wd + length_wd)].to_numpy()
np.set_printoptions(threshold=sys.maxsize)

XX_wd = X_wd
w_l_wd = math.floor(7 * 24 * 60 * 60 / time_diff)
l_wd = len(y_wd) - lag_weekly_wd * 7 - q_in_steps_wd

# getting the date series
temp_date_col_wd = pd.DatetimeIndex(df_wd.TIME).view(int) / 10 ** 9

date_range_in_days_wd = (temp_date_col_wd[
                         q_in_steps_wd + lag_weekly_wd * weekly_len_wd: int(
                             q_in_steps_wd + lag_weekly_wd * weekly_len_wd + length_wd)] -
                         temp_date_col_wd[q_in_steps_wd + lag_weekly_wd * weekly_len_wd]) / (24 * 60 * 60)


def kernel(d):
    gamma = 7
    w = np.exp(-d ** 2 / gamma)
    return w

train, test_wd = train_test_split(np.arange(0, yy_wd.size), test_size=0.2)
model_wd = KNeighborsRegressor(n_neighbors=8, weights=kernel).fit(XX_wd[train], yy_wd[train])
# model = KNeighborsRegressor(n_neighbors=5, weights='uniform').fit(XX[train], yy[train])
y_pred_wd = model_wd.predict(XX_wd[test_wd])
print(r2_score(y_true=yy_wd[test_wd], y_pred=y_pred_wd))
print(mean_squared_error(y_true=yy_wd[test_wd], y_pred=y_pred_wd))

y_pred_full_wd = model_wd.predict(XX_wd)
plot.scatter(date_range_in_days_wd, yy_wd, color="blue", label="Actual Available Bikes")
plot.scatter(date_range_in_days_wd, y_pred_full_wd, color="red", label="Predicted Available Bikes")
plot.xlabel("days")
plot.ylabel("Available Bikes")
plot.legend(bbox_to_anchor=(0.5, 0.98))
plot.show()


###########################################################################################
# Predicting Weekend Occupancies
df_we = df[((pd.DatetimeIndex(df.TIME).dayofweek == 5) | (pd.DatetimeIndex(df.TIME).dayofweek == 6))]
# Extract the required columns
y_we = df_we["AVAILABLE BIKES"]
# y = df["TIME"]

q_we = 60
lag_weekly_we = 2
# Find length of feature vector
weekly_len_we = math.floor(2 * 24 * 60 * 60 / time_diff)
q_in_steps_we = int(q_we * 60 / time_diff)
length_we = len(y_we) - lag_weekly_we * weekly_len_we - q_in_steps_we

X_we = y_we[q_in_steps_we: int(q_in_steps_we + length_we)]

for i in range(1, lag_weekly_we):
    temp = y_we[q_in_steps_we + i * weekly_len_we: int(q_in_steps_we + i * weekly_len_we + length_we)]
    X_we = np.column_stack((X_we, temp))

lag_we = 1
for i in range(lag_we - 1, -1, -1):
    temp = y_we[lag_weekly_we * weekly_len_we - i*3: int(lag_weekly_we * weekly_len_we - i*3 + length_we)]
    X_we = np.column_stack((X_we, temp))

yy_we = y_we[q_in_steps_we + lag_weekly_we * weekly_len_we: int(q_in_steps_we + lag_weekly_we * weekly_len_we + length_we)].to_numpy()
np.set_printoptions(threshold=sys.maxsize)

XX_we = X_we
temp_date_col_we = pd.DatetimeIndex(df_we.TIME).view(int) / 10 ** 9
l_we = len(y_we) - lag_weekly_we * 7 - q_in_steps_we

date_range_in_days_we = ((temp_date_col_we[
                       q_in_steps_we + lag_weekly_we * weekly_len_we: int(q_in_steps_we + lag_weekly_we * weekly_len_we + length_we)] -
                          temp_date_col_we[q_in_steps_we + lag_weekly_we * weekly_len_we]) / (24 * 60 * 60)) + 4


def kernel(d):
    gamma = 5
    w = np.exp(-d ** 2 / gamma)
    return w

train, test_we = train_test_split(np.arange(0, yy_we.size), test_size=0.2)
model_we = KNeighborsRegressor(n_neighbors=4, weights=kernel).fit(XX_we[train], yy_we[train])
# model = KNeighborsRegressor(n_neighbors=5, weights='uniform').fit(XX[train], yy[train])
y_pred_we = model_we.predict(XX_we[test_we])
print(r2_score(y_true=yy_we[test_we], y_pred=y_pred_we))
print(mean_squared_error(y_true=yy_we[test_we], y_pred=y_pred_we))
print(mean_squared_error(y_true=yy_we[test_we], y_pred=y_pred_we, squared=False))

y_pred_full_we = model_we.predict(XX_we)
plot.scatter(date_range_in_days_we, yy_we, color="blue", label="Actual Available Bikes")
plot.scatter(date_range_in_days_we, y_pred_full_we, color="red", label="Predicted Available Bikes")
plot.xlabel("days")
plot.ylabel("Available Bikes")
plot.legend(bbox_to_anchor=(0.5, 0.98))
plot.show()


# Evaluating all predictions

y_pred_all = np.concatenate((y_pred_wd, y_pred_we), axis=0)
y_actual_all = np.concatenate((yy_wd[test_wd], yy_we[test_we]), axis=0)
# print(y_pred_all)
print("=================Evaluation of final Results=================")
print(r2_score(y_true=y_actual_all, y_pred=y_pred_all))
print(mean_squared_error(y_true=y_actual_all, y_pred=y_pred_all))
# Plotting all predictions
plot.scatter(date_range_in_days_we, yy_we, color="blue", label="Actual Available Bikes")
plot.scatter(date_range_in_days_we, y_pred_full_we, color="red", label="Predicted Available Bikes")
plot.scatter(date_range_in_days_wd, yy_wd, color="blue")
plot.scatter(date_range_in_days_wd, y_pred_full_wd, color="red")
plot.xlabel("days")
plot.ylabel("Available Bikes")
plot.legend(bbox_to_anchor=(0.5, 0.98))
plot.show()
print("=====================Baseline======================")
y_pred_baseline = np.concatenate((X_wd[test_wd][:, 3], X_we[test_we][:, 2]), axis = 0)
print(r2_score(y_true=y_actual_all, y_pred=y_pred_baseline))
print(mean_squared_error(y_true=y_actual_all, y_pred=y_pred_baseline))