# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:48:28 2018

@author: zhouying
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report
import sklearn
def random_pick(some_list, probabilities):  
    import random
    x = random.uniform(0,1)  
    cumulative_probability = 0.0  
    for item, item_probability in zip(some_list, probabilities):  
        cumulative_probability += item_probability  
        if x < cumulative_probability:break  
    return item  
  
#data,label= arg
import scipy.io as scio
mydata = scio.loadmat('MNIST_data\\UCI\\ionosphere.mat')
data = np.array(mydata['data'])
label = np.squeeze(mydata['label'])
U = 1
L = 1
#deltaT = 1
h = []
hh = {}
from generalized_ir import general_ir,weighted_general_ir,boundary_gir,consin_generalized_ir
from overlapping import anove
for i in range(U):
    deltaT = 1
    while deltaT >0:
        negative = data[label==0]
        index = [i for i,x in enumerate(label) if x==0]
        Tplas,Tminas,total = weighted_general_ir([data,label])
#        Tplas,Tminas,total = anove([data,label])
        
#        Tminas+=0.1*Tm
#        total[label==0]+=0.1*to[label==0]
        total /=np.sum(total)
        deltaT = Tminas-Tplas
        reduce = random_pick(index,total[label==0])
        h.append(reduce)
        data = np.delete(data,reduce,0)
        label = np.delete(label,reduce,0)
#    dx = np.ones(label.shape[0])/(label.shape[0])
#    belta = 0
#    for j in range(L):
#        hij = KNeighborsClassifier(weights=dx)
#        hij.fit(data,label)
#        epsilon = hij.score(data,label,sample_weight=dx)
#        if epsilon<0.5:
#            continue
#        belta = epsilon/(1-epsilon)
#        pre = hij.predict(data)
#        dx[pre==label]*=belta
#        dx/=np.sum(dx)
#        h[str(j)]=hij
    hey = GaussianNB()
    hey.fit(data,label)
    hh[str(i)]=hey
    data = np.array(mydata['data'])
    label = np.squeeze(mydata['label'])

final_pre = np.zeros(label.shape[0])
pre = np.zeros(label.shape[0])
for i in range(U):
    hey = hh[str(i)]
    pre+=hey.predict(data)
pre/=U
#final_pre[pre>=0.5]=1
print(classification_report(label,pre))
import pandas as pd
h = pd.Series(h)
print(h.describe())

    
