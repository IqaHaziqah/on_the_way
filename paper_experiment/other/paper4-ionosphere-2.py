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
#import matplotlib.pyplot as plt
#from sklearn import preprocessing
#import vae
#from SDAE import mysdae
import scipy.io
from myutil2 import Smote,app,compute,write,random_walk,cross_validation,grid_search
import tensorflow as tf
#from vae4 import mnist_vae

#ionosphere yeast glass
#data=np.loadtxt('./MNIST_data/ionosphere.txt',dtype='float32')
#for windows
mydata = scipy.io.loadmat('..\\MNIST_data\\UCI\\ionosphere.mat')
#for linux
#mydata = scipy.io.loadmat('../MNIST_data/UCI/ionosphere.mat')
data = np.array(mydata['data'])
label = np.transpose(mydata['label'])
#label = np.array(mydata['label'])
label = label[0]
# Parameters for reconstruction model
para_r = {
        'learning_rate':0.001,
        'training_epochs':20,
        'batch_size':20,
        'keep_rate':0.75,
        'n_input':data.shape[1],
        'hidden_size':[25],
        'hidden_size_positive':[25],
        'scale':0.01,
        'the':1.1
        }
# parameters for the oversampling process
para_o = {
    'hidden_encoder_dim':20,                                    
    'hidden_decoder_dim':20, 
    'latent_dim':2,
    'lam':0,
    'epochs':800,
    'batch_size':35,
    'learning_rate':1e-3,
    'ran_walk':False,
    'check':False,
    'trade_off':0.5,
    'activation':tf.nn.relu,
    'optimizer':tf.train.RMSPropOptimizer,
    'norm':False,
    'decay':0.85
        }

kfold = 10
#data = preprocessing.scale(data)
#path = 'collection.xls'
pos = 1
neg = 0

i = 0
#K折交叉验证 每次跑K回
#while (i<1):
#    para_c = {'classifier':'GaussianNB','over_sampling':'None','kfold':kfold}
#    cross_validation(data,label,para_c,para_o)
#    i = i+1
#
##predict the generated samples
##random_walk = True
#i = 0
#while (i<1):
#    para_c = {'classifier':'GaussianNB','over_sampling':'random_walk','kfold':kfold}
#    cross_validation(data,label,para_c,para_o)
#    i = i+1    
##random_walk = False
#i = 0
#while (i<1):
#    para_c = {'classifier':'GaussianNB','over_sampling':'vae','kfold':kfold}
#    
#    para_o['check']=False
#    cross_validation(data,label,para_c,para_o)
#    i = i+1

from sklearn import preprocessing
##j = 0
##while(j<10):
#
from myutil2 import create_cross_validation,app2
from vae6 import mnist_vae
from sklearn.preprocessing import StandardScaler
PRE = StandardScaler
N = 10
positive = 1
result = create_cross_validation([data,label],positive,N)
generation = {}
for i in range(N):
    train,train_label,test,test_label = result[str(i)]
    
    min_max_scaler = PRE()
    train = min_max_scaler.fit_transform(train)
    test = min_max_scaler.transform(test)
    result[str(i)] = train,train_label,test,test_label
    togene = train[train_label==1]
    _,gene = mnist_vae(togene,togene.shape[0],feed_dict=para_o)
    generation[str(i)] = gene

from get import get_result
get_result(N,result,generation)


#    j = j+1

###利用SMOTE算法采样
#gF1 = []
#ggmean = []
#gauc = []  
#for i in range(N):
#    train,train_label,test,test_label = result[str(i)]
#    
#    
#    min_max_scaler = StandardScaler()
#    train = min_max_scaler.fit_transform(train)
#    test = min_max_scaler.transform(test)    
#    S = Smote(train,N=100)
#    gene = S.over_sampling()
#    train,_ = app2(train,gene)
#    train_label = np.concatenate((train_label,np.ones(gene.shape[0])),axis=0)
#    gnb = GaussianNB()
#    y_predne = gnb.fit(train,train_label).predict(test)
#    y_pro = gnb.predict_proba(test)[:,1]
#    temf,temg,tema = compute(test_label,y_predne,y_pro)
##    print('F1',temf,'AUC',tema,'gmean',temg)    
#    gF1.append(temf)
#    ggmean.append(temg)
#    gauc.append(tema)
#print('##########################zhouying###################################')
#print('mean F1:',np.mean(gF1),'mean AUC:',np.mean(gauc),'mean gmean:',np.mean(ggmean))



#from vae6 import mnist_vae
#from vae4 import wanna_see
#
##epochs = [10,20,30,40,50,60,70,80]
#epochs = [1000]
#for value in epochs:
#    para_o['epochs']=value
#    train = data[label==1]
#    para_o['check'] = True
#    
##    data = preprocessing.scale(train)   
#    min_max = PRE()
#    
#    data = min_max.fit_transform(train)
##    wanna_see(train,gene,train_label)
#    
#    z_sample,ans = mnist_vae(data,300,para_o)
#    print('trainingepoch:',value)
#    print('time used:',(time.clock()-start))

print('time used:',(time.clock()-start))
#para_c = {'classifier':'GaussianNB','over_sampling':'vae','kfold':2}    
#grid_search(data,label,para_c,para_o)
#use the reconstruction model and generated samples
    
#    print('##########################zhouying###################################')
    
#write(path,{'classifier':'GaussianNB','genesize':0,'over_sampling':'None'},
#      {'F1':F1,'AUC':auc,'gmean':gmean})
    
    
#import matplotlib.pyplot as plt
#plt.scatter(data[:,0],data[:,1],c='r')
#plt.scatter(gene[:,0],gene[:,1],c='b')
#plt.scatter(z_sample[:,0],z_sample[:,1],c='g')
