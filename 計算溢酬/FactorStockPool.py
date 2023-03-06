import pandas as pd

FactorName='Size'
Overall=pd.DataFrame()
if FactorName=='Momentum':
    # factor_6m_mom=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx','momentum1', index_col=0) 
    # factor_sue=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx','momentum2', index_col=0) 
    # factor_52high=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx','momentum3', index_col=0) 
    xl=pd.ExcelFile('./CTBCdata/因子分數20211217v1.xlsx')
    sheet_name=xl.sheet_names  # see all sheet names
    factor_6m_mom=xl.parse(sheet_name[9],header=1)
    factor_sue=xl.parse(sheet_name[10],header=1)
    factor_52high=xl.parse(sheet_name[11],header=1)
    
    validColumns=list(factor_6m_mom.columns.values)
    validColumns.remove('Ticker')
    validColumns.remove('Name')
    factor_6m_mom=factor_6m_mom.set_index('Ticker')
    for c in validColumns:
    # for c in factor_6m_mom.columns.values:
        Overall[c]=factor_6m_mom[c].values*1/3+factor_sue[c].values*1/3+factor_52high[c].values*1/3
    Overall.index=factor_6m_mom.index

elif FactorName=='Quality':
    # factor_roe=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx','quality1', index_col=0) 
    # factor_roa=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx','quality2', index_col=0) 
    # factor_gp=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx','quality3', index_col=0)
    xl=pd.ExcelFile('./CTBCdata/因子分數20211217v1.xlsx')
    sheet_name=xl.sheet_names  # see all sheet names
    factor_roe=xl.parse(sheet_name[6],header=1)
    factor_roa=xl.parse(sheet_name[7],header=1)
    factor_gp=xl.parse(sheet_name[8],header=1)
    
    validColumns=list(factor_roe.columns.values)
    validColumns.remove('Ticker')
    validColumns.remove('Name')
    factor_gp=factor_gp.set_index('Ticker')
    for c in validColumns:
        Overall[c]=factor_roe[c].values*1/3+factor_roa[c].values*1/3+factor_gp[c].values*1/3
    Overall.index=factor_gp.index

elif FactorName=='Value':
   
   
    # factor_bm=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx','因子值_價值01', index_col=0)
    # factor_ep=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx','因子值_價值02', index_col=0) 
    # factor_cfp=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx','因子值_價值03', index_col=0)
    
    xl=pd.ExcelFile('./CTBCdata/因子分數20211217v1.xlsx')
    sheet_name=xl.sheet_names  # see all sheet names
    factor_bm=xl.parse(sheet_name[2],header=1)
    factor_ep=xl.parse(sheet_name[3],header=1)
    factor_cfp=xl.parse(sheet_name[4],header=1)
    
    validColumns=list(factor_bm.columns.values)
    validColumns.remove('Ticker')
    validColumns.remove('Name')
    factor_bm=factor_bm.set_index('Ticker')
    for c in validColumns:
        Overall[c]=factor_bm[c].values*1/3+factor_ep[c].values*1/3+factor_cfp[c].values*1/3
    Overall.index=factor_bm.index
elif FactorName=='Size':
    # factor_size=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx',index_col=0) 
    # print(factor_size)
    # factor_size=pd.read_excel('./CTBCdata/因子分數20211217v1.xlsx','size', index_col=0) 
    xl=pd.ExcelFile('./CTBCdata/因子分數20211217v1.xlsx')
    sheet_name=xl.sheet_names  # see all sheet names
    factor_size=xl.parse(sheet_name[1],header=1)
    
    # print(factor_size)
    validColumns=list(factor_size.columns.values)
    validColumns.remove('Ticker')
    validColumns.remove('Name')
    factor_size=factor_size.set_index('Ticker')
    for c in validColumns:
        Overall[c]=factor_size[c].values
    Overall.index=factor_size.index
HighStockPool=pd.DataFrame()
LowStockPool=pd.DataFrame()
print(Overall)
for c in Overall.columns.values:
    NewPd=Overall[c]
    
    NewPd=NewPd.sort_values()
    NewPd=NewPd.dropna()
    AllStockNum=len(NewPd)
    HighStock=NewPd.index[:int(AllStockNum/5)].values
    LowStock=NewPd.index[int(AllStockNum*4/5):].values
 
    temp=pd.DataFrame()
    temp[c]=HighStock
    temp[c]=temp[c].astype(str)
    HighStockPool=pd.concat([HighStockPool,temp],axis=1)
    temp=pd.DataFrame()
    temp[c]=LowStock
    temp[c]=temp[c].astype(str)
    LowStockPool=pd.concat([LowStockPool,temp],axis=1)
   
HighStockPool.to_csv('./data_Processed/High_'+FactorName+'.csv')
LowStockPool.to_csv('./data_Processed/Low_'+FactorName+'.csv')
