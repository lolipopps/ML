# -*- coding: utf-8 -*-
import pandas as pd
weather_2012_final = pd.read_csv('../../data/Pandas/weather_2012.csv', index_col='Date/Time')
Temp = weather_2012_final['Temp (C)']
Tempskew = Temp.skew
Tempkurt = Temp.kurt
Tempcumsun = Temp.cumsum
Tempcummax = Temp.cummax
print(len(Temp))
unique = Temp.unique()
print(len(unique))
Temp.dropna()
Temp.drop_duplicates()
Temp.isnull
print(weather_2012_final.groupby(['Temp (C)']))
print(weather_2012_final['Temp (C)'].groupby(weather_2012_final['Temp (C)']))
#weather_2012_final.groupby('支局_维护线')['用户标识'] #上面的简单写法
#weather_2012_final.groupby('支局_维护线')['用户标识'].agg([('ADSL','count')])#按支局进行汇总对用户