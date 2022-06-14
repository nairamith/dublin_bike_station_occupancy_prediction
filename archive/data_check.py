import datetime
import math

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("../data/dublinbikes_data.csv", parse_dates=[1])
print(data.columns)

df = data[data.NAME == str('CHARLEVILLE ROAD')]

# df.groupby([pd.DatetimeIndex(df.TIME).day, pd.DatetimeIndex(df.TIME).month]).size().to_csv("../data/aggregated_results.csv")
# date_col = pd.DatetimeIndex(df.TIME).view(int) / 10 ** 9
# #date_col = pd.DatetimeIndex(df.TIME).view(int) / 10 ** 9
# time_diff = date_col[1] - date_col[0]
# plot.scatter((date_col - date_col[0]) / (24 * 60 * 60), df["AVAILABLE BIKES"], label="Actual Available Bikes")
# plot.axvline(x=28, color='r', linestyle='-', label = "Line separating days with missing data")
# plot.axvline(x=73, color='black', linestyle='-', label = "Line separating data affected with covid")
# plot.xlabel("days", fontsize=14)
# plot.ylabel("Available Bikes", fontsize=14)
# plot.legend(fontsize=14, bbox_to_anchor=(0.5, 0.98))
# plot.show()
# print(len(data))
# print(data.iloc[len(data)-1, :])

# df = df[((pd.DatetimeIndex(df.TIME) > datetime.datetime(2020, 1, 28)) &
#          (pd.DatetimeIndex(df.TIME) < datetime.datetime(2020, 2, 11)))]

date_col = pd.DatetimeIndex(df.TIME).view(int) / 10 ** 9
plot.scatter((date_col - date_col[0]) / (24 * 60 * 60), df["AVAILABLE BIKES"], label="Actual Available Bikes")
plot.xlabel("days", fontsize=14)
plot.ylabel("Available Bikes", fontsize=14)
plot.legend(fontsize=14, bbox_to_anchor=(0.7, 0.98))
plot.show()
