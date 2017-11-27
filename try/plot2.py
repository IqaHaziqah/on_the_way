# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:07:22 2017

@author: zhouying
"""

from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
    
data = pd.read_csv('.\\data\\train.csv')

xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in data['date']]
questions = data['questions']
answers = data['answers']
myid = data['id']

# 配置横坐标
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# Plot
plt.scatter(xs, questions,c='r')
plt.scatter(xs,answers,c='b')
plt.show()

ys = [d.weekday() for d in xs]
workys = [i for i in range(len(ys)) if ys[i] not in [5,6]]
restys = [i for i in range(len(ys)) if ys[i] in [5,6]]
from sklearn.linear_model import LinearRegression
import numpy as np
# 创建并拟合模型

workid = [myid[i] for i in workys]#工作日的ID号
workques=[questions[i] for i in workys] #工作日的问题数
workanws=[answers[i] for i in workys]
model1 = LinearRegression()
model1.fit(np.array(workid).reshape(len(workys),1),np.array(workques).reshape(len(workys),1))
model11 = LinearRegression()
model11.fit(np.array(workques).reshape(len(workys),1),np.array(workanws).reshape(len(workys),1))

print(model1.coef_)
print(model1.intercept_)
print(model11.coef_)
print(model11.intercept_)
restid = [myid[i] for i in restys]#周末的ID号
restques=[questions[i] for i in restys] #周末的问题数
restans = [answers[i] for i in restys]

model2 = LinearRegression()
model2.fit(np.array(restid).reshape(len(restys),1),np.array(restques).reshape(len(restys),1))
model22 = LinearRegression()
model22.fit(np.array(restques).reshape(len(restys),1),np.array(restans).reshape(len(restys),1))


print(model2.coef_)
print(model2.intercept_)
print(model22.coef_)
print(model22.intercept_)


test = pd.read_csv('.\\data\\test.csv')
testid = test['id']
testdate= [datetime.strptime(d, '%Y-%m-%d').date() for d in test['date']]
testques = []
testans = []
for i in range(len(testid)):
    if testdate[i].weekday() not in [5,6]:
        q = model1.predict(np.array(int(testid[i])).reshape(-1,1))
        testques.append(q[0][0])
        q = model11.predict(q)
        testans.append(q[0][0])
    else:
        q = model2.predict(np.array(int(testid[i])).reshape(-1,1))
        testques.append(q[0][0])
        q = model22.predict(np.array(q))
        testans.append(q[0][0])
dataframe = pd.DataFrame({'id':testid,'questions':testques,'answers':testans})
columns = ['id','questions','answers']
dataframe.to_csv('.\\data\\sample_submit3.csv',index=False,sep=',',columns=columns)
