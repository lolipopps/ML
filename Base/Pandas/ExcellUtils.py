# -*- coding: utf-8 -*-
import pandas as pd
import os
basePath = "C:\\Users\\hyt\\Desktop\\经管局\\经管局土地确权数据表\\农村土地承包经营权证发放明细表"
Files = []
def getFiles(path,fileList):
    files = os.listdir(path)
    files = [ path +"\\" + i for i in files]
    for i in files:
        if os.path.isfile(i) and (i.endswith(".xlsx") or i.endswith(".xls")):
            fileList.append(i)
        else:
            getFiles(i,fileList)
def getData(Files):
    for i in Files:
        sheets = pd.read_excel(i,sheet_name=None)
        for index,sheet in enumerate(sheets):
            sheetFile = pd.read_excel(i, sheetname=index,names=['a','b','c'])
            print(sheetFile['a'])

getFiles(basePath,Files)
getData(Files)