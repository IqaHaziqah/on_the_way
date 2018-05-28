# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import sys
sys.path.append('F:\\OneDrive\\mytensorflow\\paper_experiment')
import scipy.io as scio

#mydata = scio.loadmat('MNIST_data\\UCI\\haberman.mat')
#data = np.array(mydata['data'])
#label = np.squeeze(mydata['label'])





def compute_f1(arg):
    '''Measurement of Data Complexity for Classification Problems with Unbalanced Data'''
    '''计算论文中提到的F1'''
    data,label=arg
    positive = data[label==1]
    negative = data[label==0]
    re = []
    n = data.shape[0]
    pos = np.mean(positive,axis=0)
    si1 = np.var(positive,0)*(n/(n-1))
    neg = np.mean(negative,axis=0)
    si2 = np.var(negative,0)*(n/(n-1))
    for i in range(data.shape[1]):
        if (si1[i]+si2[i]) ==0:
            tem = 0
        else:
            tem = (pos[i]-neg[i])**2/(si1[i]+si2[i])
        re.append(tem)
    return np.max(re)

def compute_cm(arg,delta=0.025):
    '''Measurement of Data Complexity for Classification Problems with Unbalanced Data'''
    '''计算论文中提到的CM，统计难以分类的样本的平均占比，难以分类以样本周围不同类标的样本数占比决定，使用阈值决定k'''
    data,label= arg
    from sklearn.neighbors import NearestNeighbors
    k = 1
    tem = 0
    count = 0
    while True:
        nbrs = NearestNeighbors(n_neighbors=(k+1), algorithm="auto").fit(data)
        _,indices = nbrs.kneighbors(data)
        nn = indices[:,1:]
        for i in range(nn.shape[0]):
            temp = 0
            for value in label[nn[i]]:
                if value != label[i]:
                    temp += 1
            if temp >= int(k/2):
                count += 1
        count /= data.shape[0]
        if count-tem < delta:
            break
        else:
            tem = count
            k += 2
    return count

def compute_cm_fixedk(arg,k=5):
    '''Measurement of Data Complexity for Classification Problems with Unbalanced Data'''
    '''计算论文中提到的CM，统计难以分类的样本的平均占比，难以分类以样本周围不同类标的样本数占比决定,这里使用的是固定的k'''
    data,label= arg
    from sklearn.neighbors import NearestNeighbors
    count = 0
    nbrs = NearestNeighbors(n_neighbors=(k+1), algorithm="auto").fit(data)
    _,indices = nbrs.kneighbors(data)
    nn = indices[:,1:]
    for i in range(nn.shape[0]):
        temp = 0
        for value in label[nn[i]]:
            if value != label[i]:
                temp += 1
        if temp >= int(k/2):
            count += 1
    count /= data.shape[0]
    return count

#from sklearn.model_selection import StratifiedKFold
#skf = StratifiedKFold(n_splits=5)
#ans = 0.0
#for train_index,test_index in skf.split(data,label):
#    
#    ans += compute_cm_fixedk([data[train_index],label[train_index]])/5.0

#ans = compute_cm_fixedk([data,label])
