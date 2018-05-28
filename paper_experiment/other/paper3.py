#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:51:18 2017

@author: zhouying
"""
#from __future__ import division, print_function, absolute_import

import numpy as np,time
start = time.clock()
#import matplotlib.pyplot as plt
#from sklearn import preprocessing
#import vae
#from SDAE import mysdae
import scipy.io
from myutil2 import Smote,app,compute,write,random_walk,cross_validation,grid_search
#from vae4 import mnist_vae

#ionosphere yeast glass
#data=np.loadtxt('./MNIST_data/ionosphere.txt',dtype='float32')
mydata = scipy.io.loadmat('..\\MNIST_data\\UCI\\ionosphere.mat')
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
    'hidden_encoder_dim':30,                                    
    'hidden_decoder_dim':30, 
    'latent_dim':10,
    'lam':0,
    'epochs':800,
    'batch_size':30,
    'learning_rate':1e-5,
    'ran_walk':False,
    'check':True,
    'trade_off':0.5   
        }

kfold = 10
#data = preprocessing.scale(data)
#path = 'collection.xls'
pos = 1
neg = 0

i = 0
#K折交叉验证 每次跑K回
while (i<0):
    para_c = {'classifier':'GaussianNB','over_sampling':'None','kfold':10}
    cross_validation(data,label,para_c,para_o)
    i = i+1

#predict the generated samples
#random_walk = True
i = 0
while (i<0):
    para_c = {'classifier':'GaussianNB','over_sampling':'random_walk','kfold':10}
    cross_validation(data,label,para_c,para_o)
    i = i+1    
#random_walk = False
i = 0
while (i<0):
    para_c = {'classifier':'GaussianNB','over_sampling':'vae','kfold':kfold}
    para_o['ran_walk']=False
    cross_validation(data,label,para_c,para_o)
    i = i+1

from vae6 import mnist_vae
#epochs = [10,20,30,40,50,60,70,80]
epochs = [35]
for value in epochs:
    para_o['epochs']=value
    ans = mnist_vae(data,300,para_o)
    print('trainingepoch:',value)
    print('time used:',(time.clock()-start))
#para_c = {'classifier':'GaussianNB','over_sampling':'vae','kfold':2}    
#grid_search(data,label,para_c,para_o)
#use the reconstruction model and generated samples
    
#    print('##########################zhouying###################################')
    
#write(path,{'classifier':'GaussianNB','genesize':0,'over_sampling':'None'},
#      {'F1':F1,'AUC':auc,'gmean':gmean})
    
    
    



