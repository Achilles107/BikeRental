def day_of_week():
    import pandas as pd
    days=pd.DataFrame([[0,1,2,3,4,5,6],["Sun","Mon","Tue","Wed","Thr","Fri","Sat"]]).transpose()
    days.columns=['indx','dayOfWeek']
    return days
def set_day_of_week(df,days):
    df=df.merge(days,how='left',left_on='weekday',right_on='indx')
    df.drop('weekday',axis=1,inplace=True)
    return df
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
days=day_of_week()
df1=set_day_of_week(df,days)
df2=set_days(df1)
bike_scatter(df2,num_cols)
bike_box(df2,box_cols)
bike_series(df2,plt_times)
bike_hist(df2,hist_cols)
bike_hist_cond(df2,hist_cols,plt_times)