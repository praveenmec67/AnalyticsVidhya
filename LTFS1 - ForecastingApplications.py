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

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

#Preprocessing the Data:
infile=r'C:\PraveenT\PycharmProjects\Trial\Analytics Vidhya\Hackathon\LTFS\train_fwYjLYX.csv'
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
df1=pd.DataFrame(df1)
rol_mean=df1['case_count'].rolling(30).mean()
X=df1['case_count'].values
x_adf=adfuller(X)                              #From ADFuller test(test static < critical value, reject H0). Series is stationary
x_kpss=kpss(X)                                 #From KPSS test(test static > critical value, reject H0). Series is non stationary
#print(x_adf)
#print(x_kpss)


X_1=df1.diff(1).dropna()
rol_mean1=X_1['case_count'].rolling(30).mean()
x_1_adf=adfuller(X_1['case_count'].values)
x_1_kpss=kpss(X_1['case_count'].values)
#print(x_1_adf)
#print(x_1_kpss)


#Plotting to check for stationarity:
fig,ax=plt.subplots(2,2)
ax[0,0].plot(df1['case_count'].tolist())
ax[0,0].plot(rol_mean.tolist(),color='red')
ax[0,0].set_title('Raw Data(Segment_1) Case_Count Vs Rolling Mean(30)')
ax[0,1].plot(X_1['case_count'].tolist())
ax[0,1].plot(rol_mean1.tolist(),color='orange')
ax[0,1].set_title('Lag1(Segment_1) Case Count Vs Rolling Mean(30)')
tsaplots.plot_acf(X,ax[1,0])
ax[1,0].set_title('ACF for Original Series')
tsaplots.plot_acf(X_1,ax[1,1])
ax[1,1].set_title('ACF for Lag1')
#plt.show()

#Training the model:
train=df1['case_count'][:round(0.80*int(len(df1['case_count'])))]
test=df1['case_count'][round(0.80*int(len(df1['case_count']))):]
model=auto_arima(train,trace=True,error_action='ignore', suppress_warnings=True)
m=model.fit(train)
predict=model.predict(len(test))
print(test)
print(predict)

#Calculating the mean absolute error:
fin=mean_absolute_error(test,predict)
print(fin)
print(fin*len(test)/np.sum(test))
