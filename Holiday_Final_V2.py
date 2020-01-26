import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import holidays as h

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

infile=r'C:\PraveenT\PycharmProjects\Trial\Analytics Vidhya\Hackathon\LTFS\train_fwYjLYX.csv'
testfile=r'C:\PraveenT\PycharmProjects\Trial\Analytics Vidhya\Hackathon\LTFS\test_1eLl9Yf.csv'
samplefile=r'C:\PraveenT\PycharmProjects\Trial\Analytics Vidhya\Hackathon\LTFS\sample_submission_LT.csv'
holidays=r'C:\PraveenT\PycharmProjects\Trial\Analytics Vidhya\Hackathon\LTFS\Holidays.csv'
outfile=r'C:\PraveenT\PycharmProjects\Trial\Analytics Vidhya\Hackathon\LTFS\Submission_S1389101112_S29101112_HOLI_Final_V2.csv'


df = pd.read_csv(infile)


df['year'] = pd.DatetimeIndex(df['application_date']).year
df['month'] = pd.DatetimeIndex(df['application_date']).month
df['day'] = pd.DatetimeIndex(df['application_date']).day
df['dayofweek'] = pd.DatetimeIndex(df['application_date']).dayofweek
df['monthstart']= pd.DatetimeIndex(df['application_date']).is_month_start
df['monthstart']=pd.get_dummies(df['monthstart'])
df['monthend']=pd.DatetimeIndex(df['application_date']).is_month_end
df['monthend']=pd.get_dummies(df['monthend'])
df['quarterstart']=pd.DatetimeIndex(df['application_date']).is_quarter_start
df['quarterstart']=pd.get_dummies(df['quarterstart'])
df['quarterend']=pd.DatetimeIndex(df['application_date']).is_quarter_end
df['quarterend']=pd.get_dummies(df['quarterend'])
df['yearstart']=pd.DatetimeIndex(df['application_date']).is_year_start
df['yearstart']=pd.get_dummies(df['yearstart'])
df['yearend']=pd.DatetimeIndex(df['application_date']).is_year_end
df['yearend'] = pd.get_dummies(df['yearend'])
df['quarter'] = pd.DatetimeIndex(df['application_date']).quarter
df['dayofyear'] = pd.DatetimeIndex(df['application_date']).dayofyear
df['weekofyear'] = pd.DatetimeIndex(df['application_date']).weekofyear
df['festmonth']=(pd.DatetimeIndex(df['application_date']).month)


holi = pd.read_csv(infile)
holi= pd.read_csv(holidays,sep=';')
holi['DATE']=pd.to_datetime(holi['DATE'])
holi=holi.rename(columns={'DATE':'application_date'})
print(holi.columns)
fun=lambda x:x.replace("'",'')
fun1=lambda x:x.replace('/','')
holi['HOLIDAY']=holi['HOLIDAY'].apply(fun)
holi['HOLIDAY']=holi['HOLIDAY'].apply(fun1)


dict={'New Years Day':1,'Makar Sankranti  Pongal':2, 'Republic Day':3,
 'Maha Shivaratri':4, 'Holi':5, 'Ugadi  Gudi Padwa':6, 'Ram Navami':7,
 'Mahavir Jayanti':8, 'Good Friday':9, 'Labor Day':10, 'Budhha Purnima':11, 'Rath Yatra':12,
 'Eid-ul-Fitar':13, 'Raksha Bandhan':14, 'Janmashtami':15, 'Independence Day':16,
 'Vinayaka Chaturthi':17, 'Bakri Id  Eid ul-Adha':18, 'Onam':19, 'Dussehra  Dasara':20,
 'Muharram':21, 'Mathatma Gandhi Jayanti':22, 'Diwali  Deepavali':23, 'Milad un Nabi':24,
 'Christmas':25, 'Guru Nanaks Birthday':26}


df_seg_01 = df[df['segment'] == 1]
df_seg_02 = df[df['segment'] == 2]

df_seg_01['application_date'] = pd.to_datetime(df_seg_01['application_date'])
df_seg_02['application_date'] = pd.to_datetime(df_seg_02['application_date'])

df_seg_01=df_seg_01.merge(holi,how='left',on=['application_date'])
df_seg_01=df_seg_01.drop('DAY',axis=1)
df_seg_01['HOLIDAY']=df_seg_01['HOLIDAY'].map(dict)
df_seg_01['HOLIDAY']=df_seg_01['HOLIDAY'].fillna(value=0)
print(len)

df_seg_02=df_seg_02.merge(holi,how='left',on=['application_date'])
df_seg_02=df_seg_02.drop('DAY',axis=1)
df_seg_02['HOLIDAY']=df_seg_02['HOLIDAY'].map(dict)
df_seg_02['HOLIDAY']=df_seg_02['HOLIDAY'].fillna(value=0)


df_seg_01_fun1=lambda x:1 if (x==3)|(x==8)|(x==9)|(x==10)|(x==11)|(x==12) else(0)
df_seg_02_fun2=lambda x:1 if (x==9)|(x==10)|(x==11)|(x==12) else(0)

df_seg_01['festmonth']=df_seg_01['festmonth'].apply(df_seg_01_fun1)
df_seg_02['festmonth']=df_seg_02['festmonth'].apply(df_seg_02_fun2)



grouped_df_01 = df_seg_01.groupby(['year','month','day','dayofweek','quarter','dayofyear','weekofyear','monthstart','monthend','quarterstart','quarterend','yearstart','yearend','festmonth','HOLIDAY']).agg({'case_count': sum})
grouped_df_02 = df_seg_02.groupby(['year','month','day','dayofweek','quarter','dayofyear','weekofyear','monthstart','monthend','quarterstart','quarterend','yearstart','yearend','festmonth','HOLIDAY']).agg({'case_count': sum})


grouped_df_01 = grouped_df_01.reset_index()
grouped_df_02 = grouped_df_02.reset_index()

#fig = go.Figure(data=go.Scatter(x=df['application_date'],y=grouped_df_01['case_count']))
#fig.show()

index = int(round(grouped_df_01.shape[0]*0.99,1))

x_train_01 = grouped_df_01.iloc[0:index,:-1]
x_test_01 = grouped_df_01.iloc[index:,:-1]
y_train_01 = grouped_df_01.iloc[0:index,-1]
y_test_01 = grouped_df_01.iloc[index:,-1]



index = int(round(grouped_df_02.shape[0]*0.99,1))

x_train_02 = grouped_df_02.iloc[0:index,:-1]
x_test_02 = grouped_df_02.iloc[index:,:-1]
y_train_02 = grouped_df_02.iloc[0:index,-1]
y_test_02 = grouped_df_02.iloc[index:,-1]



model_01 = xgb.XGBRegressor(n_estimators=1000)
model_02 =  xgb.XGBRegressor(n_estimators=1000)

model_01.fit(x_train_01, y_train_01,
        eval_set=[(x_train_01, y_train_01), (x_test_01, y_test_01)],
        early_stopping_rounds=50,
       verbose=False)
model_02.fit(x_train_02, y_train_02,
        eval_set=[(x_train_02, y_train_02), (x_test_02, y_test_02)],
        early_stopping_rounds=50,
       verbose=False)


predicted_01 = model_01.predict(x_test_01)
print('predicted_01 len: '+str(len(predicted_01)))
predicted_02 = model_02.predict(x_test_02)


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mean_absolute_percentage_error(y_true=y_test_01,
                   y_pred= predicted_01))
print(mean_absolute_percentage_error(y_true=y_test_02,
                   y_pred= predicted_02))

sub = pd.read_csv(testfile)
sub['year'] = pd.DatetimeIndex(sub['application_date']).year
sub['month'] = pd.DatetimeIndex(sub['application_date']).month
sub['day'] = pd.DatetimeIndex(sub['application_date']).day
sub['dayofweek'] = pd.DatetimeIndex(sub['application_date']).dayofweek
sub['quarter'] = pd.DatetimeIndex(sub['application_date']).quarter
sub['dayofyear'] = pd.DatetimeIndex(sub['application_date']).dayofyear
sub['weekofyear'] = pd.DatetimeIndex(sub['application_date']).weekofyear
sub['monthstart']=pd.DatetimeIndex(sub['application_date']).is_month_start
sub['monthstart']=pd.get_dummies(sub['monthstart'])
sub['monthend']=pd.DatetimeIndex(sub['application_date']).is_month_end
sub['monthend']=pd.get_dummies(sub['monthend'])
sub['quarterstart']=pd.DatetimeIndex(sub['application_date']).is_quarter_start
sub['quarterstart']=pd.get_dummies(sub['quarterstart'])
sub['quarterend']=pd.DatetimeIndex(sub['application_date']).is_quarter_end
sub['quarterend']=pd.get_dummies(sub['quarterend'])
sub['yearstart']=pd.DatetimeIndex(sub['application_date']).is_year_start
sub['yearstart']=pd.get_dummies(sub['yearstart'])
sub['yearend']=pd.DatetimeIndex(sub['application_date']).is_year_end
sub['yearend'] = pd.get_dummies(sub['yearend'])
sub['festmonth']=(pd.DatetimeIndex(sub['application_date']).month)


sub_01 = sub[sub['segment']==1]
sub_02 = sub[sub['segment']==2]

sub_01['application_date'] = pd.to_datetime(sub_01['application_date'])
sub_02['application_date'] = pd.to_datetime(sub_02['application_date'])


sub_01=sub_01.merge(holi,how='left',on=['application_date'])
print(sub_01.head(100))

#sub_01=sub_01.iloc[:-1,:]
sub_01=sub_01.drop('DAY',axis=1)
sub_01['HOLIDAY']=sub_01['HOLIDAY'].map(dict)
sub_01['HOLIDAY']=sub_01['HOLIDAY'].fillna(value=0)

sub_02=sub_02.merge(holi,how='left',on=['application_date'])
#sub_02=sub_02.iloc[:-1,:]
sub_02=sub_02.drop('DAY',axis=1)
sub_02['HOLIDAY']=sub_02['HOLIDAY'].map(dict)
sub_02['HOLIDAY']=sub_02['HOLIDAY'].fillna(value=0)


sub01_fun1=lambda x:1 if (x==3)|(x==8)|(x==9)|(x==10)|(x==11)|(x==12) else(0)
sub02_fun2=lambda x:1 if (x==9)|(x==10)|(x==11)|(x==12) else(0)

sub_01['festmonth']=sub_01['festmonth'].apply(sub01_fun1)
sub_02['festmonth']=sub_02['festmonth'].apply(sub02_fun2)
list_01 = model_01.predict(sub_01.iloc[:,3:])
list_02 = model_02.predict(sub_02.iloc[:,3:])
final = np.append(list_01,list_02)


#rint(final[25])
#print(final[57])
#print(final[86])

#submiss = pd.read_csv(samplefile)
#submiss['case_count'] = final
#submiss.to_csv(outfile,index = False)