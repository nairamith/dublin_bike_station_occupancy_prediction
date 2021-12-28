import math

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mplot
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("../data/dublinbikes_data.csv",  parse_dates=[1])
print(data.columns)

data = data[data.NAME == str('PEARSE STREET')]

print(len(data))
date_array_in_seconds = pd.DatetimeIndex(data.iloc[:, 1]).view(np.int64)/10**9

time_gap = date_array_in_seconds[1] - date_array_in_seconds[0]

time_bins = date_array_in_seconds - date_array_in_seconds[0]

y = data.iloc[:, 6]

q = 10
lag = 3
stride = 1
w = math.floor(7*24*60*60/time_gap)
len = y.size - w - lag*w - q
XX = y[q:q+len:stride]

for i in range(1, lag):
    X = y[i*w+q:i*w+q+len:stride]
    XX = np.column_stack((XX,X))

d = math.floor(24*60*60/time_gap) # number of samples per day

for i in range(0,lag):
    X = y[i*d+q:i*d+q+len:stride]
    XX = np.column_stack((XX, X))

for i in range(0, lag):
    X = y[i:i+len:stride]
    XX = np.column_stack((XX, X))

time_in_days = time_bins/24/60/60
print(time_in_days)
yy = y[lag*w+w+q:lag*w+w+q+len:stride]

tt = time_in_days[lag*w+w+q:lag*w+w+q+len:stride]

print(XX)
print(yy)
# print(np.shape(tt))
#
#
# # model = Ridge.fit(XX, y)
# # XX = PolynomialFeatures(5).fit_transform(XX)
# X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.2)
#
# # model = Ridge(fit_intercept=False).fit(XX, yy)
# model = KNeighborsRegressor(n_neighbors=5, weights='uniform').fit( XX, yy)
# y_pred = model.predict(X_test)
#
# print(mean_squared_error(y_pred, y_test))
# # print(model.)
# y_pred_full = model.predict(XX)
# mplot.scatter(time_in_days[lag*w+w+q:lag*w+w+q+len:stride], y[lag*w+w+q:lag*w+w+q+len:stride], color = "blue")
# mplot.scatter(tt, y_pred_full, color = "red")
# # mplot.xlim((4*7,4*7+4))
#
# mplot.show()