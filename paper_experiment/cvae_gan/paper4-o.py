#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:51:18 2017

@author: zhouying
"""
#from __future__ import division, print_function, absolute_import

import numpy as np,time
start = time.clock()
import sklearn
import scipy.io
import sys
sys.path.append('..\\')
from myutil2 import Smote,app,compute,write,random_walk,cross_validation,grid_search
import tensorflow as tf
from cvaegan_v7 import VAE_GAN
import copy
import logging
logging.basicConfig(level=logging.INFO,filename='log2',filemode='w')
#from vae4 import mnist_vae

def generate(N,data,label,para_o):
    from myutil2 import create_cross_validation
    from sklearn.preprocessing import StandardScaler
    PRE = StandardScaler
    positive = 1
    result = create_cross_validation([data,label],positive,N)
    generation_1 = {}
    generation_2 = {}
    generation_3 = {}
    for i in range(N):
        train,train_label,test,test_label = result[str(i)]        
        min_max_scaler = PRE()
        train = min_max_scaler.fit_transform(train)
        test = min_max_scaler.transform(test)
        result[str(i)] = train,train_label,test,test_label
        togene = train[train_label==1]
        sess = tf.Session()
        c = VAE_GAN(sess,para_o)
        c.build_model()
        c.train([train,train_label])
        size = togene.shape[0]
        generation_1[str(i)] = c.oversampling(size)
        generation_2[str(i)] = c.oversampling(size*2)
        generation_3[str(i)] = c.oversampling(size*3)
        c.close()
        tf.reset_default_graph()
#        print('the step %d is finished'%i)
    return result,generation_1,generation_2,generation_3
# parameters for the oversampling process
para_o = {
    'latent_dim':2,
    'epochs':1000,
    'batch_size':20, #为偶数
    'learning_rate':0.005,
    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[20,10,20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[2.5,1,1,1,0.1],
    'dataset_name':'sonar'
        }
mydata = scipy.io.loadmat('..\\MNIST_data\\UCI\\'+para_o['dataset_name']+'.mat')
#for linux
#mydata = scipy.io.loadmat('../MNIST_data/UCI/ionosphere.mat')
data = np.array(mydata['data'])
label = np.squeeze(mydata['label'])#np.transpose

para_o['input_dim'] = data.shape[1]
from myutil2 import get_resultNB
M = 10
best = {}
count = 0
best['result'] = np.zeros(shape=[3,3])
result,generation_1,generation_2,generation_3 = generate(M,data,label,para_o)
ans = get_resultNB(M,result,generation_1),get_resultNB(M,result,generation_2),get_resultNB(M,result,generation_3)


#the selection for learning_rate
for value in [0.1,0.01,0.005,0.001,0.0005]:
    para_o['learning_rate']=value
    result,generation_1,generation_2,generation_3 = generate(M,data,label,para_o)
    print(value)
    ans = get_resultNB(M,result,generation_1),get_resultNB(M,result,generation_2),get_resultNB(M,result,generation_3)
    for value1,value2 in zip(ans,best['result']):
        for i in range(len(value1)):
            if value1[i]>=value2[i]:
                count+=1
    if count>len(ans)*len(ans[0])//2:
        best['result']=ans
        best['para']=copy.deepcopy(para_o)
    count = 0
    logging.warning('the para_o is %s',para_o)
    logging.warning('the answer is %s',ans)
para_o=copy.deepcopy(best['para'])
    
#the selection for lamda[0]
for value in [0.1,0.5,1,2,2.5]:
    para_o['lamda'][0]=value
    result,generation_1,generation_2,generation_3 = generate(M,data,label,para_o)
    print(value)
    ans = get_resultNB(M,result,generation_1),get_resultNB(M,result,generation_2),get_resultNB(M,result,generation_3)
    for value1,value2 in zip(ans,best['result']):
        for i in range(len(value1)):
            if value1[i]>value2[i]:
                count+=1
    if count>len(ans)*len(ans[0])//2:
        best['result']=ans
        best['para']=copy.deepcopy(para_o)
    count = 0
    logging.warning('the para_o is %s',para_o)
    logging.warning('the answer is %s',ans)
para_o=copy.deepcopy(best['para'])

#the selection for lamda[2,3]
for value in [1,0.5,0.1,0.05,0.01]:
    para_o['lamda'][2]=value
    para_o['lamda'][3]=value
    result,generation_1,generation_2,generation_3 = generate(M,data,label,para_o)
    print(value)
    ans = get_resultNB(M,result,generation_1),get_resultNB(M,result,generation_2),get_resultNB(M,result,generation_3)
    for value1,value2 in zip(ans,best['result']):
        for i in range(len(value1)):
            if value1[i]>value2[i]:
                count+=1
    if count>len(ans)*len(ans[0])//2:
        best['result']=ans
        best['para']=copy.deepcopy(para_o)
    count = 0
    logging.warning('the para_o is %s',para_o)
    logging.warning('the answer is %s',ans)    
para_o=copy.deepcopy(best['para'])

#the selection for latent_dim
for value in [4,8,10,16]:
    para_o['latent_dim']=value
    result,generation_1,generation_2,generation_3 = generate(M,data,label,para_o)
    print(value)
    ans = get_resultNB(M,result,generation_1),get_resultNB(M,result,generation_2),get_resultNB(M,result,generation_3)
    for value1,value2 in zip(ans,best['result']):
        for i in range(len(value1)):
            if value1[i]>value2[i]:
                count+=1
    if count>len(ans)*len(ans[0])//2:
        best['result']=ans
        best['para']=copy.deepcopy(para_o)
    count = 0
    logging.warning('the para_o is %s',para_o)
    logging.warning('the answer is %s',ans)    


print('time used:',(time.clock()-start))