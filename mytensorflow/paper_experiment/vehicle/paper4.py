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
mydata = scipy.io.loadmat('..\\MNIST_data\\UCI\\vehicle.mat')
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
    'epochs':350,
    'batch_size':35,
    'learning_rate':1e-3,
    'ran_walk':False,
    'check':False,
    'trade_off':0.5,
    'activation':tf.nn.relu,
    'optimizer':tf.train.AdamOptimizer,
    'norm':False,
    'decay':0.85,
    'initial':3
        }

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

def pretraining(X):
    import numpy as np
    import matplotlib.pyplot as plt
#    from sklearn.preprocessing import MinMaxScaler
#    m = MinMaxScaler()
#    X = m.fit_transform(X)
    X -= np.mean(X,axis=0)
    cov = np.dot(X.T,X)/X.shape[0]
    U,S,V = np.linalg.svd(cov)
    print(U.shape,S.shape,V.shape)
    print(S)
    plt.hist(S)
    Xrot = np.dot(X,U)
    Xwhite = Xrot/np.sqrt(S+1e-5)
    return Xwhite
def generate(N,data,label,para_o):
    from myutil2 import create_cross_validation
    from vae7 import mnist_vae   
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    PRE = MinMaxScaler()   
    positive = 1
#    data = pretraining(data)
    result = create_cross_validation([data,label],positive,10)
    generation_1 = {}
    generation_2 = {}
    generation_3 = {}
    for i in range(N):
        train,train_label,test,test_label = result[str(i)]
        train = PRE.fit_transform(train)
        test = PRE.transform(test)
        result[str(i)] = train,train_label,test,test_label
        togene = train[train_label==1]
#        _,gene_1,gene_2,gene_3 = mnist_vae(togene,togene.shape[0],feed_dict=para_o)
        gene_1,gene_2,gene_3 = mnist_vae(togene,[100,200,300],feed_dict=para_o)
        generation_1[str(i)] = gene_1
        generation_2[str(i)] = gene_2
        generation_3[str(i)] = gene_3
    return result,generation_1,generation_2,generation_3
#import pickle
#with open('result','wb') as f:
#    pickle.dump(result,f,0)    
#f.close()
#with open('generation1','wb') as f:
#    pickle.dump(generation,f,0)
#f.close()
#generation = {}
#for i in range(N):
#    train,train_label,test,test_label = result[str(i)]
#    togene = train[train_label==1]
#    _,gene = mnist_vae(togene,togene.shape[0]*2,feed_dict=para_o)
#    generation[str(i)] = gene
#with open('generation2','wb') as f:
#    pickle.dump(generation,f,0)
#f.close()
#generation = {}
#for i in range(N):
#    train,train_label,test,test_label = result[str(i)]
#    togene = train[train_label==1]
#    _,gene = mnist_vae(togene,togene.shape[0]*3,feed_dict=para_o)
#    generation[str(i)] = gene
#with open('generation3','wb') as f:
#    pickle.dump(generation,f,0)
#f.close()
#from get import get_result
#print(para_o)
from myutil2 import get_resultC,get_resultNB,get_resultNN
#e = [-4,-3,-2,-1]
a,b,c=np.zeros([3,6])
#for value in e:
p = {}
M = 2
result,generation_1,generation_2,generation_3 = generate(M,data,label,para_o)
ans=get_resultC(M,result,generation_1)
if ans>c[0]:
    c[0]=ans
    p['C7']=para_o
ans=get_resultC(M,result,generation_2)
if ans>c[2]:
    c[2]=ans
    p['C8']=para_o
ans=get_resultC(M,result,generation_3)
if ans>c[4]:
    c[4]=ans
    p['C9']=para_o

#result,generation_1,generation_2,generation_3 = generate(10,data,label,para_o)
ans = get_resultNB(M,result,generation_1)
if ans>b[0]:
    b[0]=ans
    p['NB4']=para_o
ans=get_resultNB(M,result,generation_2)
if ans>b[2]:
    b[2]=ans
    p['NB5']=para_o
ans=get_resultNB(M,result,generation_3)
if ans>b[4]:
    b[4]=ans
    p['NB6']=para_o

#result,generation_1,generation_2,generation_3 = generate(10,data,label,para_o)

ans = get_resultNN(M,result,generation_1)
if ans>a[0]:
    a[0]=ans
    p['NN1']=para_o
ans = get_resultNN(M,result,generation_2)
if ans>a[2]:
    a[2]=ans
    p['NN2']=para_o
ans = get_resultNN(M,result,generation_3)
if ans>a[4]:
    a[4]=ans
    p['NN3']=para_o
#from vae4 import mnist_vae
##from vae4 import wanna_see
##
##epochs = [10,20,30,40,50,60,70,80]
#from sklearn.preprocessing import StandardScaler
#PRE = StandardScaler
#
#train = data[label==1]
#para_o['check'] = False
#para_o['norm']=False
#para_o['epochs']=1500
##    data = preprocessing.scale(train)   
#
##    wanna_see(train,gene,train_label)
#
#origin = mnist_vae(train,300,para_o)

#    print('trainingepoch:',value)
#    print('time used:',(time.clock()-start))

print('time used:',(time.clock()-start))

