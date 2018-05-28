# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:03:04 2018

@author: zhouying
"""

def compute(index,data,label):
    negative = data[label==0]
    positive = data[label==1]
    from sklearn.neighbors import KNeighborsClassifier
    if label[index]==1:
        nei = KNeighborsClassifier(n_neighbors=1)
        nei.fit(negative,[i for i,value in enumerate(label) if value==0])
        key = nei.predict(data[index,:].reshape(1,-1))
        dis = nei.kneighbors(data[index,:].reshape(1,-1))
    else:
        nei = KNeighborsClassifier(n_neighbors=1)
        nei.fit(positive,[i for i,value in enumerate(label) if value==1])
        key = nei.predict(data[index,:].reshape(1,-1))
        dis = nei.kneighbors(data[index,:].reshape(1,-1))
#    print(key,dis)
    return key[0],dis[0][0]
import numpy as np
data = np.array([[1,1],[1,2],[2,1],[2,2],[2.5,1],[2.5,2],[3,1],[3,2]])
label = np.array([0,0,1,1,0,0,1,1])
def boundary(args):
    data,label=args
    index = [i for i in range(label.shape[0])]
    reduce = []
    thre = 1
    for i in index:
        temp = {'key':[]}
        tr = i
        while True:        
            tem,dis = compute(tr,data,label)        
            if tem in temp['key']:
                break
            else:
                temp['key'].append(tem)
                temp[str(tem)]= []
                temp[str(tem)].append(dis[0])    
            
            tr = tem
        for j,k in enumerate(temp['key']):
            if j==0:
                continue
            elif temp[str(temp['key'][j-1])]>=thre*temp[str(temp['key'][j])]:
                if temp['key'][j-1] not in reduce:
                    reduce.append(temp['key'][j-1])  
#    baoliu = [i for i in index if i not in reduce]
    return reduce

reduce = boundary([data,label])