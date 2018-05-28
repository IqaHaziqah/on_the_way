# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:29:33 2018

@author: zhouying
"""
def overlapping(arg):
    data,label = arg
    import numpy as np
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
            tem = (fenzi/fenmu)
        confusion += tem
    ov = confusion/data.shape[1]
    return ov

def weight_overlapping(arg):
    data,label = arg
    import numpy as np
    import scipy.stats as st
    pos = data[label==1]
    neg = data[label==0]
    weight = 0
    confusion = 0.0
    for i in range(data.shape[1]):
        x = data[:,i]
        weight = st.pearsonr(x,label)[0]
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
            tem = (fenzi/fenmu)*(1-weight)/2
        confusion += tem
    ov = confusion/data.shape[1]
    return ov

def another_overlapping(arg):
    data,label = arg
    import numpy as np
    import scipy.stats as st
    pos = data[label==1]
    neg = data[label==0]
    record = {}
    con = np.zeros(data.shape[0])
    for i in range(data.shape[1]):
        x = data[:,i]
        tem = st.pearsonr(x,label)[0]
        if np.isnan(tem):
            tem = 0
        record['weight'+str(i)]=(1-tem)
        x = pos[:,i]
        record['zheng'+str(i)] = [np.min(x),np.max(x)]
        x = neg[:,i]
        record['fu'+str(i)] = [np.min(x),np.max(x)]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if label[i]==1 and data[i,j]>=record['fu'+str(j)][0]and data[i,j]<=record['fu'+str(j)][1]:
                con[i]+=record['weight'+str(j)]
            elif label[i]==0 and data[i,j]>=record['zheng'+str(j)][0]and data[i,j]<=record['zheng'+str(j)][1]:
                con[i]+=record['weight'+str(j)]
    con = con/data.shape[1]
#    print(con)
    print('the another overlapping is %.3f'%(np.mean(con)))
    return np.mean(con)

def anove(arg):
    data,label = arg
    import numpy as np
    import scipy.stats as st
    pos = data[label==1]
    neg = data[label==0]
    record = {}
    con = np.zeros(data.shape[0])
    for i in range(data.shape[1]):
        x = data[:,i]
        tem = st.pearsonr(x,label)[0]
        if np.isnan(tem):
            tem = 0
        record['weight'+str(i)]=(1-tem)/2
        x = pos[:,i]
        record['zheng'+str(i)] = [np.min(x),np.max(x)]
        x = neg[:,i]
        record['fu'+str(i)] = [np.min(x),np.max(x)]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if label[i]==1 and data[i,j]>=record['fu'+str(j)][0]and data[i,j]<=record['fu'+str(j)][1]:
                con[i]+=1#record['weight'+str(j)]
            elif label[i]==0 and data[i,j]>=record['zheng'+str(j)][0]and data[i,j]<=record['zheng'+str(j)][1]:
                con[i]+=1#record['weight'+str(j)]
    con = con/data.shape[1]
#    print(con)
#    print('the another overlapping is %.3f'%(np.mean(con)))
