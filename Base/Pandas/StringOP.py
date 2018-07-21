# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['figure.figsize'] = (15, 3)
# plt.rcParams['font.family'] = 'sans-serif'
# weather_2012 = pd.read_csv('../../data/Pandas/weather_2012.csv', parse_dates=True, index_col='Date/Time')
# print(weather_2012[:5])
# weather_description = weather_2012['Weather']
# is_snowing = weather_description.str.contains('Snow')
# print(is_snowing[:5])
# print(is_snowing.astype(float)[:10])
# print(is_snowing.astype(float).resample('M').apply(np.mean))
#
# temperature = weather_2012['Temp (C)'].resample('M').apply(np.median)
# is_snowing = weather_2012['Weather'].str.contains('Snow')
# snowiness = is_snowing.astype(float).resample('M').apply(np.mean)
#
# # Name the columns
# temperature.name = "Temperature"
# snowiness.name = "Snowiness"
# print(pd.concat([temperature, snowiness], axis=1))

popcon = pd.read_csv('../../data/Pandas/popularity-contest', sep=' ', )[:-1]
popcon.columns = ['atime', 'ctime', 'package-name', 'mru-program', 'tag']
print(popcon[0:5])
popcon['atime'] = popcon['atime'].astype(int)
popcon['ctime'] = popcon['ctime'].astype(int)
popcon['atime'] = pd.to_datetime(popcon['atime'], unit='s')
popcon['ctime'] = pd.to_datetime(popcon['ctime'], unit='s')
print(popcon[0:5])