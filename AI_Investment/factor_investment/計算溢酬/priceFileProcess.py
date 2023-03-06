from numpy.core.arrayprint import dtype_is_implied
import pandas as pd
import numpy as np
from pandas.core.arrays.sparse import dtype
col_name = ['coid','stockName','date','open','high','low','close','volume_d','volume_m','return','overturn','SharesOutstanding','marketValue','最後顯示買價','最後顯示賣價','log_return','marketValueRatio','成交值比重','成交筆數','本益比-TSE','本益比-TEJ','股價淨值比-TSE','股價淨值比-TEJ','漲跌停','股價營收比-TEJ','股利殖利率-TSE','dividendRate','股價漲跌','高低價差比','次日開盤參考價','次日漲停價','次日跌停價','注意股票','處置股票','全額交割','市場別']
df = pd.read_csv('./data/2005T2021RawData.csv', header=0, names=col_name,low_memory=False)
print(df)
Pool=list(np.unique(df['coid'].values))


GNNStockPool=Pool

GNNStockPool.append('50')
GNNStockPool.append('56')
GNNStockPool.append('0050')
GNNStockPool.append('0056')

print(len(GNNStockPool))

AllStockMaxDate=[]
AllStockMinDate=[]
AllStock=[]
for chooseStock in list(np.unique(df['coid'].values)):
# for chooseStock in GNNStockPool:
    print(chooseStock)
    # print(np.unique(df['coid'].values))
    if str(chooseStock) not in GNNStockPool:
        continue

    stockFrame=df[df['coid']==str(chooseStock)]
    newDataFrame=pd.DataFrame()
    print(stockFrame)
    AllDates=[]
    MaxDate='0001-01-01'
    MinDate='9999-99-99'
    for d in list(stockFrame['date'].values):
        dateString=d.replace('/','-')
        if dateString>MaxDate:
            MaxDate=dateString
        if dateString<MinDate:
            MinDate=dateString
        AllDates.append(d.replace('/','-'))
    newDataFrame['Date']=AllDates
    newDataFrame['low_adj']=stockFrame['low'].values
    newDataFrame['high_adj']=stockFrame['high'].values
    newDataFrame['open_adj']=stockFrame['open'].values
    newDataFrame['close_adj']=stockFrame['close'].values
    newDataFrame['volume']=stockFrame['volume_m'].values
    newDataFrame['marketValue']=stockFrame['marketValue'].values
   
    newDataFrame['return']=stockFrame['return'].values
    newDataFrame['stock']=str(chooseStock)
    print(newDataFrame.head(1))
    print(newDataFrame.tail(1))
    # print(AllDates)
    
    AllStockMaxDate.append(MaxDate)
    AllStockMinDate.append(MinDate)
    AllStock.append(chooseStock)
    newDataFrame.to_csv('./data/price/'+str(chooseStock)+'.csv')

StockValidation=pd.DataFrame()
StockValidation['stock_id']=AllStock
StockValidation['date_start']=AllStockMinDate
StockValidation['date_end']=AllStockMaxDate

StockValidation.to_csv('./data/StockValidation.csv')
