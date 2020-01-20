import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.stattools import pacf,acf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.graphics import tsaplots
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import mean_absolute_error
from statsmodels.tsa.statespace.api import SARIMAX
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

#Preprocessing the Data:
infile=r'C:\PraveenT\PycharmProjects\Trial\Analytics Vidhya\Hackathon\LTFS\train_fwYjLYX.csv'
outfile=r'C:\PraveenT\PycharmProjects\Trial\Analytics Vidhya\Hackathon\LTFS\output.txt'
df=pd.read_csv(infile)
df['application_date']=pd.to_datetime(df['application_date'])
df1=df[df['segment']==1]
df2=df[df['segment']==2]
df1=df1.set_index(df1['application_date'])
df2=df2.set_index(df2['application_date'])

#Grouping the data as per the requirement:
df1=df1.groupby('segment').resample('D').case_count.sum()
df2=df2.groupby('segment').resample('D').case_count.sum()

#Checking if the given series is stationary for segement 1:
X=pd.DataFrame(df1)
#fig,ax=plt.subplots(2,2,sharex=True)
fig,ax1=plt.subplots(2,2,sharex=True)
#ax[0,0].plot([i for i in X.index.get_level_values(1)],X['case_count'].tolist())
#ax[0,0].set_title('Original Raw Data(Segment1)')
#X_adf=adfuller(X['case_count'].values)
#X_kpss=kpss(X['case_count'].values)


X_log=pd.DataFrame(np.log(X['case_count']).replace([np.inf,-np.inf],np.nan).fillna(value=np.log(X['case_count']+1)))
print(X_log)
#rol_mean=X_log['case_count'].rolling(30).mean().dropna()
#rol_std=X_log['case_count'].rolling(30).std().dropna()
#ax[0,1].plot([i for i in X_log['case_count'].index.get_level_values(1)],X_log['case_count'].tolist())
#ax[0,1].plot([i for i in rol_mean.index.get_level_values(1)],rol_mean.tolist(),color='red')
#ax[0,1].plot([i for i in rol_std.index.get_level_values(1)],rol_std.tolist  (),color='black')
#ax[0,1].set_title('Log Transform of Raw Data Vs Moving Average and Std')


#Xlr=(X_log['case_count']-rol_mean).dropna()
#xlr_r1=Xlr.rolling(30).mean().dropna()
#xlr_s1=Xlr.rolling(30).std().dropna()
#Xlr_adf=adfuller(Xlr.values)
#Xlr_kpss=kpss(Xlr.values)
#print(Xlr_adf)
#print(Xlr_kpss)


#X_log_diff=X_log.diff(1).fillna(value=0)
#X_diff=X['case_count'].diff(1).replace(np.nan,X['case_count'][0])

#print(pd.Series(sm.tsa.acf(X_1,nlags=10)))


#Training the model:
#train=X_log_diff['case_count'][:round(0.80*int(len(X_log_diff['case_count'])))]
#test=X_log_diff['case_count'][round(0.80*int(len(X_log_diff['case_count']))):]
#train=X_diff[:round(0.85*int(len(X_diff)))]
#test=X_diff[round(0.85*int(len(X_diff))):]
train=X_log
model=auto_arima(train,trace=True,error_action='ignore', suppress_warnings=True)
m=model.fit(train)
predict=m.predict(87)
print(np.exp(predict))

#a=np.asarray(X_log_diff['case_count'][:len(train)])
#a=np.asarray(X_diff[:len(train)])
#a=np.asarray(X_diff)
#a=np.asarray(X_log)
#X_log_diff_inv_inp=np.concatenate((a,predict))
#X_diff_inv_inp=np.concatenate((a,predict))
#X_inv=np.exp(X_log_diff_inv_inp.cumsum())
#X_inv=(X_diff_inv_inp.cumsum())


#predict_org=(X_inv[len(X_diff):])
#predict_org=(X[len(X):])
#test_org=np.asarray(X['case_count'][len(train):])
#print(predict_org)
#print(len(predict_org))

#Calculating the mean absolute error:
#fin=mean_absolute_error(test,predict)
#print(fin)
#a=((np.sum((np.abs(test_org-predict_org))/(test_org)))*100)/len(test)
#print(a)




