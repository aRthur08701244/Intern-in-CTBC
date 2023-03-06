import pandas as pd
import matplotlib.pyplot as plt
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
draw=AllData[AllData['Date']>='2013-04-29']
draw=draw.set_index('Date')
draw.plot()
plt.show()

# Market=pd.read_csv('./data/Y9999.csv')
# Benchmark=pd.read_csv('./data/price/0050.csv')

