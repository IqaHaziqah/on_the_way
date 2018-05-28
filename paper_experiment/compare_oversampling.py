# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:01:28 2018

@author: zhouying
"""

import sys
sys.path.append('vae')
sys.path.append('distribution_ovsampling')
import pandas as pd
import numpy as np
import scipy.io as scio
from myutil2 import create_cross_validation,get_resultNB,compute
from vae6 import mnist_vae
from ndo import normal,smote
from sklearn.naive_bayes import GaussianNB
import sklearn

'''load the dataset'''

dataset = 'ionosphere'
mydata = scio.loadmat('MNIST_data\\UCI\\'+dataset+'.mat')
data = np.array(mydata['data'])
label = np.squeeze(mydata['label'])
para_o = pd.read_pickle('vae\\'+dataset+'.txt')
f1 = open('vae.txt','ab')
f2 = open('ndo.txt','ab')
f3 = open('smo.txt','ab')


result = create_cross_validation([data,label],1,10)
for i in range(1):
    
    train,train_label,test,test_label = result[str(i)]
    ########vae
    ov_vae,_,_ = mnist_vae(train[train_label==1],train.shape[0],para_o)
    
    model = sklearn.neighbors.KNeighborsClassifier()
    model.fit(train,np.arange(0,train_label.shape[0]))#求最近邻的编号
    pre = model.predict(ov_vae)
    info_0 = len(pre[train_label[pre]==0])#生成样本中0类标的个数
    info_1 = len(pre[train_label[pre]==1])#生成样本中1类标的个数
    pre = model.predict(ov_vae)
    pre = np.array(list(set(pre)))
    dive_0 = len(pre[train_label[pre]==0])#生成样本中不同的类标0的个数
    dive_1 = len(pre[train_label[pre]==1])#生成样本中不同的类标1的个数
    
    train_1 = np.concatenate((train,ov_vae),axis=0)
    train_label1 = np.concatenate((train_label,np.ones(ov_vae.shape[0])),axis=0)    
    
    gnb = GaussianNB()
    y_predne = gnb.fit(train_1,train_label1).predict(test)
    y_pro = gnb.predict_proba(test)[:,1]
    re = compute(test_label,y_predne,y_pro)
    print(info_0,info_1,dive_0,dive_1)
    print(re)
#    np.savetxt(f1,[info_0,info_1,dive_0,dive_1],fmt='%d')
#    np.savetxt(f1,np.array([re]),fmt='%.4f')
    
    #######ndo
    ov_ndo,_,_ = normal(train,100)
#    ov_ndo,_,_ = mnist_vae(train[train_label==1],train.shape[0],para_o)
    model = sklearn.neighbors.KNeighborsClassifier()
    model.fit(train,np.arange(0,train_label.shape[0]))#求最近邻的编号
    pre = model.predict(ov_ndo)
    info_0 = len(pre[train_label[pre]==0])#生成样本中0类标的个数
    info_1 = len(pre[train_label[pre]==1])#生成样本中1类标的个数
    pre = model.predict(ov_ndo)
    pre = np.array(list(set(pre)))
    dive_0 = len(pre[train_label[pre]==0])
    dive_1 = len(pre[train_label[pre]==1])
    
    train_1 = np.concatenate((train,ov_ndo),axis=0)
    train_label1 = np.concatenate((train_label,np.ones(ov_ndo.shape[0])),axis=0)    
    
    gnb = GaussianNB()
    y_predne = gnb.fit(train_1,train_label1).predict(test)
    y_pro = gnb.predict_proba(test)[:,1]
    re = compute(test_label,y_predne,y_pro)
    print(info_0,info_1,dive_0,dive_1)
    print(re)
#    np.savetxt(f2,[info_0,info_1,dive_0,dive_1],fmt='%d')
#    np.savetxt(f2,np.array([re]),fmt='%.4f')    #get_resultNB(1,result,ov_ndo)
    
    #####smote
    ov_smo,_,_ = smote(train)
#    ov_smo,_,_ = mnist_vae(train[train_label==1],train.shape[0],para_o)
    model = sklearn.neighbors.KNeighborsClassifier()
    model.fit(train,np.arange(0,train_label.shape[0]))#求最近邻的编号
    pre = model.predict(ov_smo)
    info_0 = len(pre[train_label[pre]==0])#生成样本中0类标的个数
    info_1 = len(pre[train_label[pre]==1])#生成样本中1类标的个数
    pre = model.predict(ov_smo)
    pre = np.array(list(set(pre)))
    dive_0 = len(pre[train_label[pre]==0])
    dive_1 = len(pre[train_label[pre]==1])
    train_1 = np.concatenate((train,ov_smo),axis=0)
    train_label1 = np.concatenate((train_label,np.ones(ov_smo.shape[0])),axis=0)    
    
    gnb = GaussianNB()
    y_predne = gnb.fit(train_1,train_label1).predict(test)
    y_pro = gnb.predict_proba(test)[:,1]
    re = compute(test_label,y_predne,y_pro)
    print(info_0,info_1,dive_0,dive_1)
    print(re)
#    np.savetxt(f3,[info_0,info_1,dive_0,dive_1],fmt='%d')
#    np.savetxt(f3,np.array([re]),fmt='%.4f')
f1.close()
f2.close()
f3.close()