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
print(data.iloc[len(data)-1, :])