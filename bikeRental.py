#Adds day of week
def day_of_week():
    import pandas as pd
    days=pd.DataFrame([[0,1,2,3,4,5,6],["Sun","Mon","Tue","Wed","Thr","Fri","Sat"]]).transpose()
    days.columns=['indx','dayOfWeek']
    return days
#joins dataset and day of week
def set_day_of_week(df,days):
    df=df.merge(days,how='left',left_on='weekday',right_on='indx')
    df.drop('weekday',axis=1,inplace=True)
    return df
#Sets days in the dataset
def set_days(df):
    import pandas as pd
    df['days']=pd.Series(range(df.shape[0]))/24
    df['days']=df['days'].astype('int')
    return df
    
import pandas as pd
data=pd.read_csv("C:\Users\Achilles\Pictures\ksy\dbike.csv")
df=pd.DataFrame(data)
#df['atemp']
del(df['atemp'],df['casual'],df['registered'],df['instant'],df['dteday'])
num_cols=["temp","hum","windspeed","hr"]
def bike_scatter(df,cols):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import statsmodels.nonparametric.smoothers_lowess as lw
    
    print('Columns='+str(df.columns))
    print('cols='+str(cols))
    for col in cols:
        print(col)
        los=lw.lowess(df['cnt'],df[col], frac=0.3)
        fig=plt.figure(figsize=(8,6))
        fig.clf()
        ax=fig.gca()
        df.plot(kind='scatter',x=col,y='cnt',ax=ax)
        plt.plot(los[:,0],los[:,1],axes=ax,color='r')
        ax.set_xlabel(col)
        ax.set_ylabel('No. of Bikes')
        ax.set_title('No. of bikes vs. '+col)
        fig.savefig('scatter_'+col+'.png')
    return 'DONE'
box_cols=['season','yr','mnth','hr','holiday','workingday','weathersit','dayOfWeek']
def bike_box(df,cols):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    for col in cols:
        fig =plt.figure(figsize=(8,6))
        fig.clf()
        ax=fig.gca()
        df.boxplot(column='cnt',by=col,ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel('No of bikes')
        ax.set_title('Number of bikes vs'+col)
        fig.savefig('box_'+col+'.png')
    return 'DONE'
plt_times=[6,8,10,12,14,16,18,20]

def bike_series(df,tms):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    lims=(min(df.cnt),max(df.cnt))
    for t in tms:
        fig =plt.figure(figsize=(8,6))
        fig.clf()
        ax=fig.gca()
        df[df.hr==t].plot(kind='line',x='days',y='cnt',ylim=lims,ax=ax)
        plt.xlabel("Days from start ")
        plt.ylabel("Bikes rented")
        plt.title("Bikes rented by day for hour="+str(t))
        fig.savefig('series_'+str(t)+'.png')
    return 'DONE'
hist_cols=["cnt","temp","hum","windspeed"]
def bike_hist(df,cols):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    
    for col in cols:
        fig=plt.figure(figsize=(8,6))
        fig.clf()
        ax=fig.gca()
        df[col].hist(bins=30,ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Density of '+col)
        ax.set_ylabel('Density of '+col)
        fig.savefig('hist_'+col+'.png')
    return 'DONE'
#Display histograms by hrs
def bike_hist_cond(df,col,by):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(nrows=len(by),ncols=1,figsize=(8,6))
    for i,t in enumerate(by):
        temp=df.ix[df.hr==t,col]
        ax[i].hist(temp.as_matrix(),bins=30)
        ax[i].set_title('Bikes rented at time = '+str(t))
    fig.savefig('hist_')
#displaying bikes rented with time series
def ts_bikes(df,times):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    for tm in times:
        fig=plt.figure(figsize=(8,6))
        fig.clf()
        ax=fig.gca()
        df[df.hr == tm].plot(kind='line',x='days',y='cnt',ax=ax)
        df[df.hr == tm].plot(kind='line',x='days',y='prediction',color='red',ax=ax)
        plt.xlabel("Days from start")
        plt.ylabel("Number of bikes rented")
        plt.title("Bikes rented for hour="+str(tm))
        fig.savefig('ts_'+str(tm)+'.png')
    return 'DONE'
#Calculating residuals
def residuals(df):
    df['resids']=df.prediction - df.cnt
    return df
#Displaying residuals
def box_residuals(df):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    df=residuals(df)
    fig=plt.figure(figsize=(8,6))
    fig.clf()
    ax=fig.gca()
    df.boxplot(column=['resids'],by=['hr'],ax=ax)
    plt.xlabel('')
    plt.ylabel('Residuals')
    plt.title("Boxplot grouped by hr")
    fig.savefig('boxes'+'.png')
    return 'DONE'
#Displaying residuals with time series
def ts_resids_hist(df,times):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    for tm in times:
        fig=plt.figure(figsize=(8,6))
        fig.clf()
        ax=fig.gca()
        temp=df.ix[df.hr==tm,'resids']
        ax.hist(temp.as_matrix(),bins=30)
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title("Residuals for hour="+str(tm))
        fig.savefig('hist_'+str(tm)+'.png')
    return 'DONE'
days=day_of_week()
df1=set_day_of_week(df,days)
df2=set_days(df1)

#Part 1 (Without separating working hours and non working hours) 
train_data=df2.head(12165)
test_data=df2.tail(5214)
test_data1=df2.tail(5214)
train_labels=train_data['cnt']
del(train_data['cnt'],train_data['dayOfWeek'])
train_features=train_data
from sklearn import datasets, linear_model
#Linear Regression model
reg=linear_model.LinearRegression()
reg.fit(train_features,train_labels)
del(test_data1['cnt'],test_data1['dayOfWeek'])
import numpy as np
prediction=[]
for i in np.arange(len(test_data)):
    pred_val=test_data1.iloc[[i]]
    #Predicting on testing data
    l=reg.predict(pred_val)
    l=' '.join(map(str,l))
    coef=reg.coef_
    prediction.append(l)
test_data['prediction']=prediction
test_data['prediction']=pd.to_numeric(test_data['prediction'],errors='coerce')
times=[6,8,10,12,14,16,18,20,22]
ts_bikes(test_data,times)
residuals(test_data)
box_residuals(test_data)
ts_resids_hist(test_data,times)


#Part 2 (Separating working hours and non working hours)
#Spliting working hours and non working hours
def is_Working(df):
    import numpy as np
    work_day=df['workingday'].as_matrix()
    holiday=df['holiday'].as_matrix()
    df['isWorking']=np.where(np.logical_and(work_day==1, holiday==0),1,0)
    df['monthCount']=12*df.yr+df.mnth
    isWorking=df['isWorking'].as_matrix()
    df['workHr']=np.where(isWorking,df.hr,df.hr+24.0)
    df=df.drop(['workingday','holiday','hr'], axis=1)
    return df
def lower_quantile(df):
    import pandas as pd
    out=df.groupby(['monthCount','workHr']).cnt.quantile(q=0.2)
    out=pd.DataFrame(out)
    out.reset_index(inplace=True)
    out.columns=['monthCount','workHr','quantile']
    return out
def quantile_2(df,quantile):
    import pandas as pd
    #Save original names of dataframe
    in_names=list(df)
    df=pd.merge(df,quantile,left_on=['monthCount','workHr'],right_on=['monthCount','workHr'],how='inner')
    #Filter rows
    df=df.ix[df['cnt']>df['quantile']]
    #Remove unwanted cols
    df.drop('quantile',axis=1,inplace=True)
    df.columns=in_names
    #Sort data based on dayCount
    df.sort(['days','workHr'], axis=0,inplace=True)
    return df
#refer ts_bikes()
def ts_bikes1(df,times):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    for tm in times:
        fig=plt.figure(figsize=(8,6))
        fig.clf()
        ax=fig.gca()
        df[df.workHr == tm].plot(kind='line',x='days',y='cnt',ax=ax)
        df[df.workHr == tm].plot(kind='line',x='days',y='prediction',color='red',ax=ax)
        plt.xlabel("Days from start")
        plt.ylabel("Number of bikes rented")
        plt.title("Bikes rented for hour="+str(tm))
        fig.savefig('ts1_'+str(tm)+'.png')
    return 'DONE'
#refer box_residuals()
def box_residuals1(df):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    df=residuals(df)
    fig=plt.figure(figsize=(8,6))
    fig.clf()
    ax=fig.gca()
    df.boxplot(column=['resids'],by=['workHr'],ax=ax)
    plt.xlabel('')
    plt.ylabel('Residuals')
    plt.title("Boxplot grouped by workingHr")
    fig.savefig('boxes1'+'.png')
    return 'DONE'
#refer ts_resids_hist()
def ts_resids_hist1(df,times):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    for tm in times:
        fig=plt.figure(figsize=(8,6))
        fig.clf()
        ax=fig.gca()
        temp=df.ix[df.workHr==tm,'resids']
        ax.hist(temp.as_matrix(),bins=30)
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title("Residuals for workHour="+str(tm))
        fig.savefig('hist1_'+str(tm)+'.png')
    return 'DONE'
#Calculating working day
df3=is_Working(df2)
#Calculating quantile where quantile=0.2
quantile=lower_quantile(df3)
#merging df3 and quantile
df4=quantile_2(df3,quantile)
#Spliting the data
train_features1=df4.head(9287)
test_data3=df4.tail(3980)
test_data2=df4.tail(3980)
train_labels1=train_features1['cnt']
del(train_features1['cnt'],train_features1['dayOfWeek'])
#train
#Linear Regression model
reg1=linear_model.LinearRegression()
reg1.fit(train_features1,train_labels1)
del(test_data2['cnt'],test_data2['dayOfWeek'])
import numpy as np
prediction1=[]
for i in np.arange(len(test_data2)):
    pred_val1=test_data2.iloc[[i]]
    #Predicting on testing data
    l1=reg1.predict(pred_val1)
    l1=' '.join(map(str,l1))
    coef=reg1.coef_
    prediction1.append(l1)
test_data3['prediction']=prediction1
test_data3['prediction']=pd.to_numeric(test_data3['prediction'],errors='coerce')
times1=[6,8,10,12,14,16,18,20,22,30,32,34,36,38,40]
ts_bikes1(test_data3,times1)
residuals(test_data3)
box_residuals1(test_data3)
ts_resids_hist1(test_data3,times1)
