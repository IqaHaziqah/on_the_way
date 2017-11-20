# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:07:22 2017

@author: zhouying
"""

from datetime import datetime

import csv
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# 生成横纵坐标信息
#with open( '.\\data\\train.csv','rt') as f:
#    reader = csv.DictReader(f)
#    dates = [row['date'] for row in reader]    
#f.close()
#with open( '.\\data\\train.csv','rt') as f:
#    reader = csv.DictReader(f)
#    questions = [row['questions'] for row in reader]    
#f.close()
#with open( '.\\data\\train.csv','rt') as f:
#    reader = csv.DictReader(f)
#    answers = [row['answers'] for row in reader]    
#f.close()
#    
#with open( '.\\data\\train.csv','rt') as f:
#    reader = csv.DictReader(f)
#    myid = [row['id'] for row in reader]    
#f.close()
    

xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

# 配置横坐标
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# Plot
plt.scatter(xs, questions,c='r')
plt.scatter(xs,answers,c='b')
plt.show()

#ys = [d.weekday() for d in xs]
#workys = [i for i in range(len(ys)) if ys[i] not in [5,6]]
#restys = [i for i in range(len(ys)) if ys[i] in [5,6]]
from sklearn.linear_model import LinearRegression
import numpy as np
# 创建并拟合模型

workid = [myid[i] for i in workys]#工作日的ID号
workques=[questions[i] for i in workys] #工作日的问题数
model1 = LinearRegression()

model1.fit(np.array(workid).reshape(len(workys),1),np.array(workques).reshape(len(workys),1))


restid = [myid[i] for i in restys]#工作日的ID号
restques=[questions[i] for i in restys] #工作日的问题数
model2 = LinearRegression()
model2.fit(np.array(restid).reshape(len(restys),1),np.array(restques).reshape(len(restys),1))

model3 = LinearRegression()
model3.fit(np.array(questions).reshape(len(questions),1),np.array(answers).reshape(len(answers),1))

#with open( '.\\data\\test.csv','rt') as f:
#    reader = csv.DictReader(f)
##    testid = [row['id'] for row in reader]
#    testdate = [row['date']for row in reader]
#f.close()
testdate=[d.weekday() for d in testdate]
testques = []
testans = []
for i in range(len(testid)):
    if testdate[i].weekday() not in [5,6]:
        q = model1.predict(np.array(int(testid[i])).reshape(-1,1)[0])
        testques.append(q)
    else:
        q = model2.predict(np.array(int(testid[i])).reshape(-1,1)[0])
        testques.append(q)
    a = model3.predict(np.array(q).reshape(-1,1)[0])
    testans.append(a)