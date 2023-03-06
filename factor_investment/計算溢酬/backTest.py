
##流通市值權重＋sum(因子強度Ｘ因子權重Ｘ增益比例)
##增益比例**（排名(i,j)-因子內平均排名)/(sum(abs(排名(i,j)-因子內平均排名))/2)

import pandas as pd
import numpy as np
from pandas.core.arrays.sparse import dtype
from model_v1 import MinRiskByHistoricalVar
import matplotlib.pyplot as plt
import ffn
import dask.dataframe

candidatePoolDF=pd.read_excel('./CTBCdata/10個因子原始資料.xlsx','125檔', dtype=str,index_col=0,header=1) 
NoBenchmarkPool=list(candidatePoolDF.index)
candidatePool=['0050']+NoBenchmarkPool
candidatePool=[str(x) for x in candidatePool]
factor_size=pd.read_excel('./CTBCdata/10個因子原始資料.xlsx','因子值_SIZE', index_col=0) 



AllStockReturn=pd.DataFrame()
for stock in candidatePool:
    data=pd.read_csv('./data/price/'+str(stock)+'.csv')

    if stock==candidatePool[0]:
        AllStockReturn=data[['Date','return']]
        AllStockReturn=AllStockReturn.rename(columns={'return':str(stock)})
    else:
        AllStockReturn=AllStockReturn.merge(data[['Date','return']],on='Date',how='left')
        AllStockReturn=AllStockReturn.rename(columns={'return':str(stock)})
AllStockReturn=AllStockReturn[AllStockReturn['Date']>'2013-04-30']
dates=AllStockReturn['Date'].values
def marketValueWeightWithConstrain(factor_size,pool,adjustdate):
   
    newPD=pd.DataFrame()
    for c in factor_size.columns.values:
        newPD[str(str(c).split('T')[0])]=factor_size[c]
        # factor_size=factor_size.rename(columns={c:str(c).split('T')[0]})
    
    chooseFactorDate=newPD.columns.values[np.where(np.array(newPD.columns.values)<=adjustdate)[0][-1]]
    
    NowSizeDF=newPD[chooseFactorDate]
    NowSizeDF=NowSizeDF.reset_index()

    # print(adjustdate,NowSizeDF)
    # print(len(pool))
    NowChoosedPool=NowSizeDF[NowSizeDF['index'].isin(pool)]
    NowChoosedPool=NowChoosedPool.sort_values(by=chooseFactorDate)
    NowChoosedPool=NowChoosedPool.dropna()
    # print(NowChoosedPool)
    Big50Pool=NowChoosedPool.tail(50)
    SumAllBig50Weight=np.sum(Big50Pool[chooseFactorDate].values)
    Big50Pool['initial_weight']=Big50Pool[chooseFactorDate].values/SumAllBig50Weight
    ###Adjust Because of constrain
    AdjustWeight=[]
    ResWeight=[]
    for iw in range(len(Big50Pool['initial_weight'].values)):
        if iw<len(Big50Pool['initial_weight'].values)-1:
            
            Adjust=min(Big50Pool['initial_weight'].values[iw],0.2)
            ResWeight.append(Big50Pool['initial_weight'].values[iw]-Adjust)
            AdjustWeight.append(Adjust)

        else:

            Adjust=min(max(Big50Pool['initial_weight'].values[iw],0.2),0.3)
            ResWeight.append(Big50Pool['initial_weight'].values[iw]-Adjust)
            AdjustWeight.append(Adjust)

    
    Big50Pool['Adjust_weight']=AdjustWeight
    Big50Pool['ResWeight']=ResWeight
    Big50Pool['NoAdjustLabel']=Big50Pool['ResWeight']==0
    SumNoAdjustedWeight=sum(Big50Pool[Big50Pool['ResWeight']==0]['initial_weight'])
    Big50Pool['AdjustRatio']=np.zeros(len(Big50Pool))
    Big50Pool['AdjustRatio']=Big50Pool['initial_weight']/SumNoAdjustedWeight
    Big50Pool['AddWeight']=Big50Pool['NoAdjustLabel']*Big50Pool['AdjustRatio']*sum(ResWeight)
    Big50Pool['FinalWeight']=Big50Pool['Adjust_weight']+Big50Pool['AddWeight']
  
    Big50Pool=Big50Pool[['index','FinalWeight']]
    # print(Big50Pool)
    NewBigPool=pd.DataFrame()
    for i in range(len(Big50Pool)):
      
        NewBigPool[str(Big50Pool['index'].values[i])]=[Big50Pool['FinalWeight'].values[i]]

   
    
    NoWeightStocks=list(set(pool)-set(NewBigPool.columns.values))
    validWeightStocks=list(NewBigPool.columns.values)
    return NewBigPool,NoWeightStocks,validWeightStocks


def marketValueWeightWithoutConstrain(factor_size,pool,adjustdate):
   
    newPD=pd.DataFrame()
    for c in factor_size.columns.values:
        newPD[str(str(c).split('T')[0])]=factor_size[c]
        # factor_size=factor_size.rename(columns={c:str(c).split('T')[0]})
    chooseFactorDate=newPD.columns.values[np.where(np.array(newPD.columns.values)<=adjustdate)[0][-1]]
    
    NowSizeDF=newPD[chooseFactorDate]
    NowSizeDF=NowSizeDF.reset_index()

    # print(adjustdate,NowSizeDF)
    # print(len(pool))
    NowChoosedPool=NowSizeDF[NowSizeDF['index'].isin(pool)]
    NowChoosedPool=NowChoosedPool.sort_values(by=chooseFactorDate)
    NowChoosedPool=NowChoosedPool.dropna()
    # print(NowChoosedPool)
    Big50Pool=NowChoosedPool.tail(50)
    SumAllBig50Weight=np.sum(Big50Pool[chooseFactorDate].values)
   
    
    Big50Pool['FinalWeight']=Big50Pool[chooseFactorDate].values/SumAllBig50Weight
  
    Big50Pool=Big50Pool[['index','FinalWeight']]
    # print(Big50Pool)
    NewBigPool=pd.DataFrame()
    for i in range(len(Big50Pool)):
      
        NewBigPool[str(Big50Pool['index'].values[i])]=[Big50Pool['FinalWeight'].values[i]]
    NoWeightStocks=list(set(pool)-set(NewBigPool.columns.values))
    validWeightStocks=list(NewBigPool.columns.values)
    return NewBigPool,NoWeightStocks,validWeightStocks
c_ori=AllStockReturn.columns
AllStockReturn=AllStockReturn.reset_index()
AllStockReturn=AllStockReturn[c_ori]
AllStockWeight=AllStockReturn[AllStockReturn.columns]

AllStockReturn['portfolioReturn']=0
AllStockReturn['portfolioReturnWithCost']=0
AllStockReturn[candidatePool]=AllStockReturn[candidatePool]/100
for c in NoBenchmarkPool:
    AllStockWeight[str(c)]=np.zeros(len(AllStockWeight))

firstAdjust=False
previousWeight=pd.DataFrame()
weightDF=pd.DataFrame([np.zeros(len(NoBenchmarkPool))],columns=NoBenchmarkPool)

for idate in range(len(dates)):
    date=dates[idate]
    value=0
    changePool=False
    adjustMonthes=['01','04','07','10']
    
    if idate<len(dates)-1:
        if date.split('-')[1]!=dates[idate+1].split('-')[1]: ##end of month
            if date.split('-')[1] in adjustMonthes:
                changePool=True
                firstAdjust=True
    if firstAdjust==False:
        BeginDate=date
    if changePool:
        previousWeight=weightDF

        ##Without Constrain
        weightDF,NoWeightStocks,validWeightStocks=marketValueWeightWithoutConstrain(factor_size,NoBenchmarkPool,date)
        
        ##With Constrain
        # weightDF,NoWeightStocks,validWeightStocks=marketValueWeightWithConstrain(factor_size,NoBenchmarkPool,date)
       
        weightDF[NoWeightStocks]=0
        weightDF=weightDF.fillna(0)
        previousWeight=previousWeight.fillna(0)
        CostFrame=previousWeight.subtract(weightDF)
        
        SumAllCost=np.nansum(abs(CostFrame.values))
        EnhanceWeight=MinRiskByHistoricalVar(validWeightStocks,date)
        EnhanceWeight=EnhanceWeight.fillna(0)
      
        weightDF=pd.concat([weightDF,EnhanceWeight])
       
        Final=weightDF.sum()
        
        minPositive=np.min(Final[Final>0].values)
        Final[Final<0]=minPositive/2
       
        FinalSum=np.sum(Final.values)
        Final=Final/FinalSum
       
        weightDF=pd.DataFrame()
        for i in range(len(validWeightStocks)):
          
            weightDF[validWeightStocks[i]]=[Final[validWeightStocks[i]]]
       
        
    if firstAdjust:
        NowWeightDF=weightDF
        NowWeightDF['Date']=date
        NowReturnDF=AllStockReturn[AllStockReturn['Date']==date]
        
        NowReturnDF=pd.concat([NowReturnDF,NowWeightDF])
        NowReturnDF=NowReturnDF.set_index('Date')
      
        NowReturnDF_temp=NowReturnDF
       
        NowReturnDF_temp=NowReturnDF.dropna(axis=1)
        NowReturnDF_temp=pd.DataFrame(NowReturnDF_temp.apply(np.prod , axis=0))
        
       
        NowReturnDF_temp=NowReturnDF_temp.transpose()
        
        NowReturnDF=pd.concat([NowReturnDF,NowReturnDF_temp])
    
        allStocks=list(NowReturnDF_temp.columns)
       
        # allStocks.remove('0050')
        portReturn=np.nansum(NowReturnDF_temp[allStocks].values[0])

        if changePool==True:
            portReturnCost=portReturn-SumAllCost*0.003
        else:
            portReturnCost=portReturn
        Index=AllStockReturn.index[AllStockReturn['Date'] == date].tolist()
        AllStockReturn.iloc[Index, AllStockReturn.columns.get_loc('portfolioReturn')]=portReturn
        AllStockReturn.iloc[Index, AllStockReturn.columns.get_loc('portfolioReturnWithCost')]=portReturnCost

        # print(AllStockReturn.iloc[Index])

    changePool=False
    

Result=AllStockReturn[['Date','0050','portfolioReturn','portfolioReturnWithCost']]
Result=Result[Result['Date']>BeginDate]
Result=Result.set_index('Date')
Result.index = pd.DatetimeIndex(Result.index)
Result['Rolling_0050']=Result['0050'].rolling(252).apply(lambda x: np.prod(1 + x) - 1)
Result['Rolling_portfolioReturn']=Result['portfolioReturn'].rolling(252).apply(lambda x: np.prod(1 + x) - 1)
Result['Rolling_portfolioReturnWithCost']=Result['portfolioReturnWithCost'].rolling(252).apply(lambda x: np.prod(1 + x) - 1)



Result['0050_Addone']=Result['0050']+1
Result['portfolioReturn_Addone']=Result['portfolioReturn']+1
Result['portfolioReturnWithCost_Addone']=Result['portfolioReturn']+1


# Result['Acc_0050']=Result['0050'].map(lambda x: np.prod(1 + x) - 1)
Result['Acc_0050']=Result['0050_Addone'].cumprod()
# Result['Acc_portfolioReturn']=Result['portfolioReturn'].apply(lambda x: np.prod(1 + x) - 1)
Result['Acc_portfolioReturn']=Result['portfolioReturn_Addone'].cumprod()
Result['Acc_portfolioReturnWithCost']=Result['portfolioReturnWithCost_Addone'].cumprod()



result = Result.calc_stats()
print(f"total return [bench_fund, our_fund,our_fund_cost,]: [{result['Acc_0050'].total_return*100:.2f}%,  {result['Acc_portfolioReturnWithCost'].total_return*100:.2f}%] ")
print(f"annual return [bench_fund, our_fund,our_fund_cost,our_fund_cost_insurance]: [{result['Acc_0050'].cagr*100:.2f}%,  {result['Acc_portfolioReturnWithCost'].cagr*100:.2f}%] ")
print(f"daily sharpe [bench_fund, our_fund,our_fund_cost,our_fund_cost_insurance]: [{result['Acc_0050'].daily_sharpe:.2f},  {result['Acc_portfolioReturnWithCost'].daily_sharpe:.2f}] ")
print(f"max drawdown [bench_fund, our_fund,our_fund_cost,our_fund_cost_insurance]: [{result['Acc_0050'].max_drawdown*100:.2f}%,  {result['Acc_portfolioReturnWithCost'].max_drawdown*100:.2f}%] ")
print(f"daily_sortino [bench_fund, our_fund,our_fund_cost,our_fund_cost_insurance]: [{result['Acc_0050'].daily_sortino*100:.2f}%,  {result['Acc_portfolioReturnWithCost'].daily_sortino*100:.2f}%] ")
print(f"daily_vol [bench_fund, our_fund,our_fund_cost,our_fund_cost_insurance]: [{result['Acc_0050'].daily_vol*100:.2f}%,  {result['Acc_portfolioReturnWithCost'].daily_vol*100:.2f}%] ")
print(f"daily_skew [bench_fund, our_fund,our_fund_cost,our_fund_cost_insurance]: [{result['Acc_0050'].daily_skew*100:.2f}%,  {result['Acc_portfolioReturnWithCost'].daily_skew*100:.2f}%] ")
print(f"daily_kurt [bench_fund, our_fund,our_fund_cost,our_fund_cost_insurance]: [{result['Acc_0050'].daily_kurt*100:.2f}%,  {result['Acc_portfolioReturnWithCost'].daily_kurt*100:.2f}%] ")
print(f"calmar [bench_fund, our_fund,our_fund_cost,our_fund_cost_insurance]: [{result['Acc_0050'].calmar*100:.2f}%,  {result['Acc_portfolioReturnWithCost'].calmar*100:.2f}%] ")



plt.subplot(2,1,1)
plt.plot(Result['Rolling_0050'].values)
plt.plot(Result['Rolling_portfolioReturn'].values)
plt.plot(Result['Rolling_portfolioReturnWithCost'].values)
plt.subplot(2,1,2)
plt.plot(Result['Acc_0050'].values)
plt.plot(Result['Acc_portfolioReturn'].values)
plt.plot(Result['Acc_portfolioReturnWithCost'].values)
plt.savefig('performance')

# Result['Rolling_0050']=pd.rolling_apply(AllStockReturn[['0050']], 252, lambda x: np.prod(1 + x) - 1)
# Result['portfolioReturn']=pd.rolling_apply(AllStockReturn[['portfolioReturn']], 252, lambda x: np.prod(1 + x) - 1)
print(Result)
