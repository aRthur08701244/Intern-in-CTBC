import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
FactorName='Momentum'
highGroup=pd.read_csv('./data_Processed/High_'+FactorName+'.csv',dtype=str)
lowGroup=pd.read_csv('./data_Processed/Low_'+FactorName+'.csv',dtype=str)
StockIndex=pd.read_csv('./data/Y9999.csv')

##case:all stocks
# validStockFrame=pd.read_csv('./data/StockValidation.csv')
# allstockspool=validStockFrame['stock_id'].values
##case: ctbc stocks
allstockspool=pd.read_csv('./CTBCdata/ctbcPool.csv',dtype=str)
allstockspool=allstockspool['pool'].values
print(allstockspool)
dates=StockIndex['Date'].values
##
allStockFrame=pd.DataFrame()
for istock in range(len(allstockspool)):
    stock=allstockspool[istock]
    print(istock,'/',len(allstockspool),' processed')
    if len(stock)==4:
        data=pd.read_csv('./data/price/'+stock+'.csv')
        data=data[['Date','return','marketValue']]
        data=data.rename(columns={'marketValue':stock+'_marketValue','return':stock+'_return'})
        
        if stock==allstockspool[0]:
            allStockFrame=data
        else:
            allStockFrame=allStockFrame.merge(data,on='Date',how='outer')


print(allStockFrame)
##
poolDates=list(highGroup.columns.values)
Values=[]
FactorFrame=pd.DataFrame()
AccValue=1
AccValues=[]
for idate in range(len(dates)):
    date=dates[idate]
   
    value=0
    changePool=False
    
    adjustMonthes=['03','05','08','11']  ##資料是3/5/8/11月
    if idate<len(dates)-1:     
        if date.split('-')[1]!=dates[idate+1].split('-')[1]: ##end of month
            if date.split('-')[1] in adjustMonthes:
                changePool=True
    if len(np.where(np.array(poolDates)<date)[0])>0:
        if changePool:
            poolDate=poolDates[np.where(np.array(poolDates)<date)[0][-1]]
        
            highPool=highGroup[poolDate]
            highPool=highPool.dropna()
            highPool=highPool.values
            lowPool=lowGroup[poolDate]
            lowPool=lowPool.dropna()
            lowPool=lowPool.values
            
            HighStockFrame=pd.DataFrame()
            LowStockFrame=pd.DataFrame()
            highPoolFeatures=['Date']
            lowPoolFeatures=['Date']

            for stock in highPool:
                highPoolFeatures.append(stock+'_marketValue')
                highPoolFeatures.append(stock+'_return')

            
            for stock in lowPool:
                lowPoolFeatures.append(stock+'_marketValue')
                lowPoolFeatures.append(stock+'_return')
        
        if date>='2013-04-30':
            # print(date)
            LowStockFrame=allStockFrame[lowPoolFeatures]
            HighStockFrame=allStockFrame[highPoolFeatures]
            NowHighStockFrame=HighStockFrame[HighStockFrame['Date']==date]
            NowLowStockFrame=LowStockFrame[LowStockFrame['Date']==date]
            
            AllHighStockReturns=[]
            AllHighStockMarketValues=[]
            AllLowStockReturns=[]
            AllLowStockMarketValues=[]
            # print('NowHighStockFrame',NowHighStockFrame)
            # print('NowLowStockFrame',NowLowStockFrame)
            for stock in highPool:
                if NowHighStockFrame[stock+'_return'].values[0]==np.nan:
                    AllHighStockMarketValues.append(0)
                    AllHighStockReturns.append(0)
                elif NowHighStockFrame[stock+'_marketValue'].values[0]==np.nan:
                    AllHighStockMarketValues.append(0)
                    AllHighStockReturns.append(0)
                else:
                    AllHighStockMarketValues.append(NowHighStockFrame[stock+'_marketValue'].values[0])
                    AllHighStockReturns.append(NowHighStockFrame[stock+'_return'].values[0])
            
            for stock in lowPool:
                # print(NowLowStockFrame[stock+'_return'].values[0])
                if NowLowStockFrame[stock+'_return'].values[0]==np.nan:
                    AllLowStockMarketValues.append(0)
                    AllLowStockReturns.append(0)
                elif NowLowStockFrame[stock+'_marketValue'].values[0]==np.nan:
                    AllLowStockMarketValues.append(0)
                    AllLowStockReturns.append(0)
                else:
                    AllLowStockMarketValues.append(NowLowStockFrame[stock+'_marketValue'].values[0])
                    AllLowStockReturns.append(NowLowStockFrame[stock+'_return'].values[0])
        
            SumAllHighStock=np.nansum(AllHighStockMarketValues)
            SumAllLowStock=np.nansum(AllLowStockMarketValues)
           
            AllHighStockMarketValues=AllHighStockMarketValues/SumAllHighStock
            AllLowStockMarketValues=AllLowStockMarketValues/SumAllLowStock
            
            ### Method1:市值加權平均報酬
            # valueHigh=np.nansum(np.multiply(np.array(AllHighStockMarketValues),np.array(AllHighStockReturns)))/100
            # valueLow=np.nansum(np.multiply(np.array(AllLowStockMarketValues),np.array(AllLowStockReturns)))/100
            
            ### Method2:簡單平均報酬
            valueHigh=np.nanmean(np.array(AllHighStockReturns))/100
            valueLow=np.nanmean(np.array(AllLowStockReturns))/100
            

            
            value=valueHigh-valueLow
            AccValue=AccValue*(1+value)
            print('---',date,value,AccValue,'---')
    Values.append(value)
    AccValues.append(AccValue)

FactorFrame['Date']=dates
FactorFrame['DailyReturn']=Values
FactorFrame['AccReturn']=AccValues
FactorFrame.to_csv('./FactorValues/'+FactorName+'.csv')
plt.plot(dates,AccValues)
plt.show()




        


    
    
       
    




