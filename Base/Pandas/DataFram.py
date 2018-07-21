# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt


# This is necessary to show lots of columns in pandas 0.12.
# Not necessary in pandas 0.13.
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

# plt.rcParams['figure.figsize'] = (15, 5)
# complaints = pd.read_csv('../../data/Pandas/311-service-requests.csv')
# print(complaints.describe())  #描述性统计
# print(complaints.dtypes);  #查看各行的数据格式
# print(complaints[:5])  ## 全部的 0-4 行
# print(complaints['Complaint Type'][0:5])  ## 查看  Complaint Type 列的 0-4行
# print(complaints[:5]['Complaint Type'])  ## 这种效率高点
# print(complaints[0:5][['Complaint Type', 'Borough']]) ## 多列的0-4行
# print(complaints['Complaint Type'].value_counts())  ## 相当于 group by  Complaint Type
# complaint_counts = complaints['Complaint Type'].value_counts()
# print(complaint_counts[:10].plot(kind='bar'))
# print((complaints['Complaint Type'] == "Noise - Street/Sidewalk")[0:4])  ## 判断  Complaint Type 列是否等于 Noise - Street/Sidewalk
# is_noise = complaints['Complaint Type'] == "Noise - Street/Sidewalk"
# in_brooklyn = complaints['Borough'] == "BROOKLYN"
# print(complaints[is_noise & in_brooklyn][:5])  ## 同时满足 is_noise 和 in_brooklyn
# print(complaints['Complaint Type'].value_counts())
# print(complaints['Complaint Type'].drop_duplicates().value_counts()) #剔除重复行数据
# print(complaints.ix[1:3,[1,3]])
# complaints.ix[1:3,[1,3]]=1           #所选位置数据替换为1
# print(complaints.ix[1:3,[1,3]])

# alist = ['NYPD']
# print(complaints['Agency'][0:10])
# print(complaints['Agency'].isin(alist)[0:10])  #将要过滤的数据放入字典中


# bikes = pd.read_csv('../../data/Pandas/bikes.csv', sep=';', encoding='latin1', parse_dates=['Date'], dayfirst=True, index_col='Date')
# bikes['Berri 1'].plot()
# berri_bikes = bikes[['Berri 1']].copy()
# print(berri_bikes[0:5])
# print(berri_bikes.index[0:5])
# print(berri_bikes.index.day[0:5])
# print("----------------------")
# print(berri_bikes.columns)
# berri_bikes.loc[:,'weekday'] = berri_bikes.index.weekday  ## 增加一列 数据扩展用
# print(berri_bikes.columns)
# print(berri_bikes.groupby('weekday').aggregate(sum))  ## group by



weather_2012_final = pd.read_csv('../../data/Pandas/weather_2012.csv', index_col='Date/Time')
weather_2012_final['Temp (C)'].plot(figsize=(15, 6))
# url_template = "http://climate.weather.gc.ca/climateData/bulkdata_e.html?format=csv&stationID=5415&Year={year}&Month={month}&timeframe=1&submit=Download+Data"
# url = url_template.format(month=3, year=2012)
# weather_mar2012 = pd.read_csv(url, skiprows=15, index_col='Date/Time', parse_dates=True, encoding='latin1', header=True)
print(weather_2012_final.columns)
weather_mar20121 = weather_2012_final.dropna(axis=1, how='any')
print(weather_mar20121[0:5])
weather_mar20122 = weather_2012_final.drop(['Rel Hum (%)', 'Visibility (km)'], axis=1)
print(weather_mar20122[0:5])
pd.Series([1,2,3])
