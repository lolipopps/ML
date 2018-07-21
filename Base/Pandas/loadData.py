# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
# plt.rcParams['figure.figsize'] = (15, 5)
# fixed_df = pd.read_csv('../../data/Pandas/bikes.csv', sep=';', encoding='latin1', dayfirst=True, index_col='Date')
# ## 默认是 , 分割  names : 结果的列名列表   header : 作为列名的行号   index_col  用作行索引的列编号或者列名  prefix : 在没有列标题时
# print(fixed_df[:3])
# print(fixed_df['Berri 1'])  ## 查看某一列
# fixed_df['Berri 1'].plot()
# fixed_df.plot(figsize=(15, 10))
# ### 还有 read_table  默认 \t 分割  read_fwf

con = sqlite3.connect("../../data/Pandas/weather_2012.sqlite")
df = pd.read_sql("SELECT * from weather_2012 LIMIT 3", con)
print(df)
df = pd.read_sql("SELECT * from weather_2012 LIMIT 3", con, index_col='id')
print(df)


weather_df = pd.read_csv('../../data/Pandas/weather_2012.csv')
con = sqlite3.connect("../../data/Pandas/test_db.sqlite")
con.execute("DROP TABLE IF EXISTS weather_2012")
weather_df.to_sql("weather_2012", con)