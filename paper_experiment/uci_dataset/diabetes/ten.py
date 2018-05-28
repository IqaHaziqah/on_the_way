# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:18:14 2018

@author: zhouying
"""
def generate(N,data,label,para_o):
    from myutil2 import create_cross_validation
    from vae4 import mnist_vae
    from sklearn.preprocessing import StandardScaler
    PRE = StandardScaler
    N = 10
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
        gene_1,gene_2,gene_3 = mnist_vae(togene,togene.shape[0],feed_dict=para_o)
        generation_1[str(i)] = gene_1
        generation_2[str(i)] = gene_2
        generation_3[str(i)] = gene_3
    return result,generation_1,generation_2,generation_3
def ten_fold(NN=1):
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
    import pickle
    #from vae4 import mnist_vae
    
    #ionosphere yeast glass
    #data=np.loadtxt('./MNIST_data/ionosphere.txt',dtype='float32')
    #for windows
    mydata = scipy.io.loadmat('..\\MNIST_data\\UCI\\diabetes.mat')
    #for linux
    #mydata = scipy.io.loadmat('../MNIST_data/UCI/ionosphere.mat')
    data = np.array(mydata['data'])
    label = np.transpose(mydata['label'])
    
    #label = np.array(mydata['label'])
    label = np.squeeze(label)
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
    'hidden_encoder_dim':8,                                    
    'hidden_decoder_dim':8, 
    'latent_dim':2,
    'lam':0,
    'epochs':1000,
    'batch_size':5,
    'learning_rate':1*1e-2,
    'ran_walk':False,
    'check':False,
    'trade_off':0.5,
    'activation':tf.nn.relu,
    'optimizer':tf.train.AdamOptimizer,
    'norm':True,
    'decay':0.9,
    'initial':3
        }
    
    kfold = 10
    #data = preprocessing.scale(data)
    #path = 'collection.xls'
    pos = 1
    neg = 0
    
    i = 0
    result,gene1,gene2,gene3 = generate(10,data,label,para_o)
    return result,gene1,gene2,gene3  #K折交叉验证 每次跑K回
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
    


import sys
import numpy as np
a=[]
b = []
c = []
sys.path.append('..\\')
from get import get_result
for i in range(1):    
    result,gene1,gene2,gene3 = ten_fold(1)
    a.append(get_result(10,result,gene1))
    b.append(get_result(10,result,gene2))
    c.append(get_result(10,result,gene3))
print(np.mean(a,axis=0))
print(np.mean(b,axis=0))
print(np.mean(c,axis=0))
