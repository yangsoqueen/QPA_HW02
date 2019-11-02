import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import *

#read the data
HSI_daily_prices=pd.read_excel('HSI.xlsx')
HSI_daily_prices['Date']=pd.to_datetime(HSI_daily_prices['Date'])
HSI_daily_prices=HSI_daily_prices.set_index('Date')
#linear interpolate
HSI_daily_prices=HSI_daily_prices.interpolate()

#resample进行时间序列重采样
HSI_monthly_prices=HSI_daily_prices.resample('m').last()

#算return
HSI_monthly_prices_return=HSI_monthly_prices.pct_change()
HSI_monthly_prices_return=HSI_monthly_prices_return.dropna(how='all')

HSI_monthly_prices_average_return = np.array(HSI_monthly_prices_return.mean())
HSI_monthly_cov = np.array(HSI_monthly_prices_return.cov())  #直接用np.cov()会导致协方差矩阵存在空值？
unit_vector = np.ones(68)

A = np.dot(np.dot(HSI_monthly_prices_average_return.T, inv(HSI_monthly_cov)), HSI_monthly_prices_average_return)
B = np.dot(np.dot(HSI_monthly_prices_average_return.T, inv(HSI_monthly_cov)), unit_vector)
C = np.dot(np.dot(unit_vector.T, inv(HSI_monthly_cov)), unit_vector)

up_list = np.arange(0.005,0.105,0.005)
lb_list = []
gama_list = []
std_list = []
w_list = [] #record the weights for each up
for up in up_list:
    lb = (up * C - B)/(A * C - B * B)
    lb_list.append(lb)
    gama = (A - B * up)/(A * C - B * B)
    gama_list.append(gama)
    alpha = lb * np.dot(inv(HSI_monthly_cov), HSI_monthly_prices_average_return) + gama * np.dot(inv(HSI_monthly_cov), unit_vector)
    w_list.append(alpha)
    std = np.sqrt(((C * (up - B / C) ** 2) / (A * C - B ** 2) )+ 1 / C)
    std_list.append(std)

plt.plot(std_list, up_list, color='red',label = 'Efficient Frontier without Risk-free Assets')
plt.xlabel('STD')
plt.ylabel('Expected Return')


#加入risk-free assets
rf = 0.02/12
std_list2 = []
w_list2 = []

for up in up_list:
    k = np.sqrt(np.dot(np.dot((HSI_monthly_prices_average_return.T-unit_vector.T*rf),inv(HSI_monthly_cov)),(HSI_monthly_prices_average_return-rf*unit_vector)))
    std = (up - rf) / k
    std_list2.append(std)
    w = np.dot(np.dot(((up - rf) / (np.dot(np.dot((HSI_monthly_prices_average_return - rf * unit_vector), inv(HSI_monthly_cov)), (HSI_monthly_prices_average_return - rf * unit_vector)))), inv(HSI_monthly_cov)), (HSI_monthly_prices_average_return - rf * unit_vector))
    w_list2.append(w)


plt.plot(std_list2, up_list, color='blue',label = 'Efficient Frontier with Risk-free Assets')
plt.legend()
plt.show()

'''
fig = plt.figure()
x = up_list
ax1 = fig.add_subplot(111)
ax1.plot(std_list, color='red', linewidth=2, label='Efficient Frontier')

plt.legend(loc=4)

ax2 = ax1.twinx()
ax2.plot(std_list2, color='blue', linewidth=2, label='With Risk-Free Asset')

plt.title('Sharp Ratio Model')
plt.legend(loc=2)
plt.show()
'''
w = pd.DataFrame(w_list,index=np.arange(0.005,0.105,0.005))
w2 = pd.DataFrame(w_list2,index=np.arange(0.005,0.105,0.005))

w.to_csv('weights_withoutRf.csv')
w2.to_csv('weights_withRf.csv')









