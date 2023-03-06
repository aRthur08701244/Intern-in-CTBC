import pandas as pd
import numpy as np
factor_52high=pd.read_excel('./CTBCdata/10個因子原始資料.xlsx','因子值_52high', index_col=0) 
stocks=factor_52high.index.values
frame=pd.DataFrame()
frame['pool']=stocks
frame.to_csv('./CTBCdata/ctbcPool.csv')
print(factor_52high)