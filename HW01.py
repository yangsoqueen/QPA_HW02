import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

#read the data
HSI_daily_prices=pd.read_excel('HSI.xlsx')
HSI_daily_prices['Date']=pd.to_datetime(HSI_daily_prices['Date'])
HSI_daily_prices=HSI_daily_prices.set_index('Date')
#linear interpolate
HSI_daily_prices=HSI_daily_prices.interpolate()

#resample进行时间序列重采样
HSI_weekly_prices=HSI_daily_prices.resample('w-FRI').last()
HSI_monthly_prices=HSI_daily_prices.resample('m').last()

#算return
HSI_daily_prices_return=HSI_daily_prices.pct_change()
HSI_daily_prices_return=HSI_daily_prices_return.dropna(how='all')
HSI_weekly_prices_return=HSI_weekly_prices.pct_change()
HSI_weekly_prices_return=HSI_weekly_prices_return.dropna(how='all')
HSI_monthly_prices_return=HSI_monthly_prices.pct_change()
HSI_monthly_prices_return=HSI_monthly_prices_return.dropna(how='all')

#calculate the covariance matrix of weekly return and save as .csv
HSI_weekly_prices_return_covariance=HSI_weekly_prices_return.cov()
HSI_weekly_prices_return_covariance.to_csv('covHSI.csv')

# plot histogram for Tencent
HSI_daily_prices_return_tencent=HSI_daily_prices_return['700 HK']
HSI_daily_prices_return_tencent=HSI_daily_prices_return_tencent.dropna()
plt.hist(HSI_daily_prices_return_tencent,bins=100,density='True')
plt.title('700 HK',fontsize=10)
plt.show()

#利用 “stats” from “scipy” 对 HSI每一个 weekly returns做假设检验, n = stats.normaltest(d)
#map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
#lambda 函数相当于一个匿名函数，顾名思义就是不用取名字的函数，相当于现实中的匿名信。
#需要清洗数据,删除NAN
s=[]
HSI_weekly_prices_return_array=np.array(HSI_weekly_prices_return.T)
HSI_weekly_prices_return_list=HSI_weekly_prices_return_array.tolist()
#delete nan in list
def remove_nan_list(list):
    list = [x for x in list if ~np.isnan(x)]
    return list
s=list(map(lambda x: remove_nan_list(x),HSI_weekly_prices_return_list))
#get p_value_list by using nomaltest() function
nomaltest_list = list(map(lambda x: stats.normaltest(x),s))
nomaltest_dataframe=pd.DataFrame(nomaltest_list)
p_value=nomaltest_dataframe['pvalue']
#plot histogram
plt.hist(p_value)
plt.title('p value',fontsize=10)
plt.show()

#autocorrelation test
autocorrelation_test_list = list(map(lambda x: acorr_ljungbox(x, lags=[1,2,3,4,5]),s))
autocorrelation_test_dataframe=pd.DataFrame(autocorrelation_test_list)
#get p_value
autocorrelation_p_value=autocorrelation_test_dataframe.iloc[:,1:2]
autocorrelation_p_value_array=np.array(autocorrelation_p_value)
#make array to a list 每一行的数据从array变成list再转成dataframe
p=list(map(lambda i: autocorrelation_p_value_array[i,0].tolist(), range(68)))
autocorrelation_p_value_dataframe=pd.DataFrame(p)
#get the pp to plot the p_value for each lag
#iloc will get dataframe loc get series

#plot p_value in 一张图 两种plot的方法 第一种画出来的图更好看
pp=list(map(lambda i: autocorrelation_p_value_dataframe.loc[:,i], range(5)))
a = pd.DataFrame(pp,columns=['lag 1','lag 2','lag 3','lag 4','lag 5'])
plt.hist(pp,label=a.columns)
plt.title('p value',fontsize=10)
plt.legend() #使label奏效
plt.show()
#plt.hist(pp[0],color='blue',bins=100,label='lag 1')
#plt.hist(pp[1],color='red',bins=100,label='lag 2')
#plt.hist(pp[2],color='orange',bins=100,label='lag 3')
#plt.hist(pp[3],color='yellow',bins=100,label='lag 4')
plt.hist(pp[4],color='green',bins=100,label='lag 5')
plt.title('p value',fontsize=10)
plt.legend() #使label奏效
plt.show()

# calculate the standard deviation of daily returns
monthly_standard_deviation=HSI_daily_prices_return.resample('m').std()
yearly_standard_deviation=HSI_daily_prices_return.resample('y').std()
#standard_deviation = pd.merge(monthly_standard_deviation, yearly_standard_deviation, how='left') 合并两张表
#concat是拼接两张表
standard_deviation = pd.concat([monthly_standard_deviation, yearly_standard_deviation], axis=0)
standard_deviation.to_csv('HSI_vol.csv ')







