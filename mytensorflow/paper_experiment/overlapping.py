# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:29:33 2018

@author: zhouying
"""
def overlapping(arg):
    data,label = arg
    import numpy as np
#    import scipy.io as scio
    
#    mydata = scio.loadmat('..\\MNIST_data\\UCI\\wpbc.mat')
#    data = np.array(mydata['data'])
#    label = np.array(mydata['label'])
#    if label.shape[0]==1:
#        label = label.squeeze(0)
#    elif label.shape[1]==1:
#        label = label.squeeze(1)
    
    pos = data[label==1]
    neg = data[label==0]
    confusion = 0.0
    for i in range(data.shape[1]):
        x = data[:,i]
        total = [np.min(x),np.max(x)]
        x = pos[:,i]
        zheng = [np.min(x),np.max(x)]
        x = neg[:,i]
        fu = [np.min(x),np.max(x)]
        value1 = 0
        value2 = 0
        mix = []
        x = pos[:,i]
        for j in x:
            if j>=fu[0] and j<=fu[1]:
                mix.append(j)
                value1 +=1
            else:
                continue
        x = neg[:,i]
        for j in x:
            if j>=zheng[0] and j<=zheng[1]:
                mix.append(j)
                value2 +=1
            else:
                continue
        fenzi = np.max(mix)-np.min(mix)
        fenmu = total[1]-total[0]
        tem = 0
        x = np.min([value1,value2]),np.max([value1,value2])
        if fenmu !=0:
            tem = (fenzi/fenmu)*(x[0]/x[1])
        confusion += tem
    ov = confusion/data.shape[1]
    print(confusion)    
    print("the overlapping is %.3f" %ov)
    return ov
    #print(confusion)