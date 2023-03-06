import pandas as pd
import cvxpy as cvx
import numpy as np
from datetime import datetime
def get_optimal_weights(covariance_returns, index_weights, scale=2.0):
    
    assert len(covariance_returns.shape) == 2
    assert len(index_weights.shape) == 1
    assert covariance_returns.shape[0] == covariance_returns.shape[1]  == index_weights.shape[0]

    
    num_of_weights = len(index_weights)
    
    
    x = cvx.Variable(num_of_weights)
    
   
    portfolio_variance = cvx.quad_form(x, covariance_returns)
    
   
    objective = cvx.Minimize(portfolio_variance)
    
    
    constraints = [x >= 0, sum(x) == 1]

   
    problem = cvx.Problem(objective, constraints)
    problem.solve()
    
    x_values = x.value
    return x_values

def filterFactorDataFrame(frame,adjustDate,pool,factorName):
    Timecolumns=list(frame.columns.values)
    frame=frame.fillna(frame.median()) 
    Temp=Timecolumns
    for k in list(frame.columns.values):
        if type(k)==str:
            Timecolumns.remove(k)
   
    frame=frame.rename(columns={x:x.strftime("%Y-%m-%d") for x in Timecolumns})
    
    choosedColumnDates=frame.columns.values[np.where(frame.columns.values<adjustDate)[0][-1]]
    
    frame['Ticker']=frame['Ticker'].astype(str)
    choose_factorsize=frame[['Ticker',choosedColumnDates]]
    choose_factorsize=choose_factorsize[choose_factorsize['Ticker'].isin(pool)]
    choose_factorsize=choose_factorsize.rename(columns={choosedColumnDates:factorName})
    # print(choose_factorsize)
    return choose_factorsize

def factorsToEnhanceRatio(factors,name):
    factors['TempRank']=factors[name].rank(ascending=True)
    factors[name+'_enhanceRatio']=(factors['TempRank']-np.nanmean(factors['TempRank'].values))/(sum(abs(factors['TempRank']-np.mean(factors['TempRank'].values)))/2)
    
    return factors



def calculateStockWeightByScore(factors_Weight,pool,adjustDate):
    
    xl=pd.ExcelFile('./CTBCdata/因子分數20211217v1.xlsx')
    sheet_name=xl.sheet_names  # see all sheet names
    AllFactors=pd.DataFrame()
    
    #size
    factor_size=xl.parse(sheet_name[1],header=1)
    
    choosed_factor_size=filterFactorDataFrame(factor_size,adjustDate,pool,'size')
    # print('choosed_factor_size',choosed_factor_size)
    AllFactors=choosed_factor_size
    #value1
    factor_value1=xl.parse(sheet_name[2],header=1)
    choosed_factor_value1=filterFactorDataFrame(factor_value1,adjustDate,pool,'value1')
    AllFactors=AllFactors.merge(choosed_factor_value1,on='Ticker',how='left')
    #value2
    factor_value2=xl.parse(sheet_name[3],header=1)
    choosed_factor_value2=filterFactorDataFrame(factor_value2,adjustDate,pool,'value2')
    AllFactors=AllFactors.merge(choosed_factor_value2,on='Ticker',how='left')
    #value3
    factor_value3=xl.parse(sheet_name[4],header=1)
    choosed_factor_value3=filterFactorDataFrame(factor_value3,adjustDate,pool,'value3')
    AllFactors=AllFactors.merge(choosed_factor_value3,on='Ticker',how='left')
    #growth
    # factor_growth=xl.parse(sheet_name[5],header=1)
    # choosed_factor_growth=filterFactorDataFrame(factor_growth,adjustDate,pool,'growth')
    # AllFactors=AllFactors.merge(choosed_factor_growth,on='Ticker',how='left')
    #quality1
    factor_quality1=xl.parse(sheet_name[6],header=1)
    choosed_factor_quality1=filterFactorDataFrame(factor_quality1,adjustDate,pool,'quality1')
    AllFactors=AllFactors.merge(choosed_factor_quality1,on='Ticker',how='left')
    #quality2
    factor_quality2=xl.parse(sheet_name[7],header=1)
    choosed_factor_quality2=filterFactorDataFrame(factor_quality2,adjustDate,pool,'quality2')
    AllFactors=AllFactors.merge(choosed_factor_quality2,on='Ticker',how='left')
    #quality3
    factor_quality3=xl.parse(sheet_name[8],header=1)
    choosed_factor_quality3=filterFactorDataFrame(factor_quality3,adjustDate,pool,'quality3')
    AllFactors=AllFactors.merge(choosed_factor_quality3,on='Ticker',how='left')
    #mom1
    factor_momentum1=xl.parse(sheet_name[9],header=1)
    choosed_factor_momentum1=filterFactorDataFrame(factor_momentum1,adjustDate,pool,'momentum1')
    AllFactors=AllFactors.merge(choosed_factor_momentum1,on='Ticker',how='left')
    #mom2
    factor_momentum2=xl.parse(sheet_name[10],header=1)
    choosed_factor_momentum2=filterFactorDataFrame(factor_momentum2,adjustDate,pool,'momentum2')
    AllFactors=AllFactors.merge(choosed_factor_momentum2,on='Ticker',how='left')
    #mom3
    factor_momentum3=xl.parse(sheet_name[11],header=1)
    choosed_factor_momentum3=filterFactorDataFrame(factor_momentum3,adjustDate,pool,'momentum3')
    AllFactors=AllFactors.merge(choosed_factor_momentum3,on='Ticker',how='left')

    # print(AllFactors)
    # AllFactors['MOMScore']=(AllFactors['momentum1']+AllFactors['momentum2']+AllFactors['momentum3'])/3
    # AllFactors['VALUEScore']=(AllFactors['value1']+AllFactors['value2']+AllFactors['value3'])/3
    # AllFactors['QUALITYScore']=(AllFactors['quality1']+AllFactors['quality2']+AllFactors['quality3'])/3
    # AllFactors['SIZEScore']=AllFactors['size']
    AllFactors['MOMWeight']=factors_Weight['Momentum'].values[0]
    AllFactors['VALUEWeight']=factors_Weight['Value'].values[0]
    AllFactors['SIZEWeight']=factors_Weight['Size'].values[0]
    AllFactors['QUALITYWeight']=factors_Weight['Quality'].values[0]

    # AllFactors['FinalScore']=AllFactors['MOMWeight']*AllFactors['MOMScore']+AllFactors['VALUEWeight']*AllFactors['VALUEScore']+AllFactors['SIZEWeight']*AllFactors['SIZEScore']+AllFactors['QUALITYWeight']*AllFactors['QUALITYScore']
    
    # AllFactors['momentum1_w']=AllFactors['MOMWeight']*(1/3)*
    AllFactors=factorsToEnhanceRatio(AllFactors,'momentum1')
    AllFactors=factorsToEnhanceRatio(AllFactors,'momentum2')
    AllFactors=factorsToEnhanceRatio(AllFactors,'momentum3')
    AllFactors=factorsToEnhanceRatio(AllFactors,'value1')
    AllFactors=factorsToEnhanceRatio(AllFactors,'value2')
    AllFactors=factorsToEnhanceRatio(AllFactors,'value3')
    AllFactors=factorsToEnhanceRatio(AllFactors,'quality1')
    AllFactors=factorsToEnhanceRatio(AllFactors,'quality2')
    AllFactors=factorsToEnhanceRatio(AllFactors,'quality3')
    AllFactors=factorsToEnhanceRatio(AllFactors,'size')
    # print(AllFactors)
    AllFactors['EnhanceWeight']=AllFactors['size_enhanceRatio']*1*AllFactors['SIZEWeight']+\
                                AllFactors['quality1_enhanceRatio']*(1/3)*AllFactors['QUALITYWeight']+\
                                AllFactors['quality2_enhanceRatio']*(1/3)*AllFactors['QUALITYWeight']+\
                                AllFactors['quality3_enhanceRatio']*(1/3)*AllFactors['QUALITYWeight']+\
                                AllFactors['value1_enhanceRatio']*(1/3)*AllFactors['VALUEWeight']+\
                                AllFactors['value2_enhanceRatio']*(1/3)*AllFactors['VALUEWeight']+\
                                AllFactors['value3_enhanceRatio']*(1/3)*AllFactors['VALUEWeight']+\
                                AllFactors['momentum1_enhanceRatio']*(1/3)*AllFactors['MOMWeight']+\
                                AllFactors['momentum2_enhanceRatio']*(1/3)*AllFactors['MOMWeight']+\
                                AllFactors['momentum3_enhanceRatio']*(1/3)*AllFactors['MOMWeight']
    return AllFactors[['Ticker','EnhanceWeight']]

def MinRiskByHistoricalVar(pool,adjustDate):
    Momentum=pd.read_csv('./FactorValues/Momentum.csv')
    Quality=pd.read_csv('./FactorValues/Quality.csv')
    Value=pd.read_csv('./FactorValues/Value.csv')
    Size=pd.read_csv('./FactorValues/Size.csv')

    AllData=pd.DataFrame()
    AllData['Date']=Momentum['Date'].values
    AllData['Momentum']=Momentum['AccReturn'].values
    AllData['Quality']=Quality['AccReturn'].values
    AllData['Value']=Value['AccReturn'].values
    AllData['Size']=Size['AccReturn'].values
    AllData=AllData[AllData['Date']>='2013-04-29']

    LastYear=int(adjustDate.split('-')[0])-1
    Month=adjustDate.split('-')[1]
    Day=adjustDate.split('-')[2]
    AllData=AllData[AllData['Date']<=adjustDate]
    AllData=AllData[AllData['Date']>=str(LastYear)+'-'+Month+'-'+Day]
    AllData=AllData.set_index('Date')
    
  
    factorweights=get_optimal_weights(AllData.cov(),np.zeros(4))
    factorweights=pd.DataFrame([factorweights],columns=['Momentum','Quality','Value','Size'])

  
    AllFactors=calculateStockWeightByScore(factorweights,pool,adjustDate)
    
    ##Transpose
    Final=pd.DataFrame()
    for i in AllFactors['Ticker'].values:
        choose=AllFactors[AllFactors['Ticker']==str(i)]
        # print(choose)
        Final[i]=choose['EnhanceWeight'].values
      
    return Final


# MinRiskByHistoricalVar(['0050','1101','1102','1216','2330','2317','6505','2412','2454'],'2013-09-29')

