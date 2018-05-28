# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:18:14 2018

@author: zhouying
"""
def ten_fold(name,para_o=[]):
    import numpy as np
    import scipy.io
    from myutil2 import Smote,app,compute,write,random_walk,cross_validation,grid_search
    import tensorflow as tf
    import pandas as pd
    filepath = '..\\MNIST_data\\UCI\\'+name+'.mat'
    mydata = scipy.io.loadmat(filepath)
    data = np.array(mydata['data'])
    label = np.transpose(mydata['label'])
    label = np.squeeze(label)
    if para_o==[]:
        para_o = pd.read_pickle(name+'.txt')
    from myutil2 import create_cross_validation
    from vae6 import mnist_vae    
    PRE = para_o['PRE']
    N = 10
    positive = 1
    result = create_cross_validation([data,label],positive,N)
    generation_1 = {}
    generation_2 = {}
    generation_3 = {}
    for i in range(N):
        train,train_label,test,test_label = result[str(i)]        
        if PRE!=[]:
            min_max_scaler = PRE()
            train = min_max_scaler.fit_transform(train)
            test = min_max_scaler.transform(test)
        result[str(i)] = train,train_label,test,test_label
        togene = train[train_label==1]
        gene_1,gene_2,gene_3 = mnist_vae(togene,togene.shape[0],feed_dict=para_o)
        generation_1[str(i)] = gene_1
        generation_2[str(i)] = gene_2
        generation_3[str(i)] = gene_3    
    
    return result,generation_1,generation_2,generation_3  #K折交叉验证 每次跑K回


def final(name,fold,para_o=[]):    
    import numpy as np
    a = []
    b = []
    c = []
    from get import get_result
    for i in range(fold):    
        result,gene1,gene2,gene3 = ten_fold(name,para_o)
        a.append(get_result(10,result,gene1))
        b.append(get_result(10,result,gene2))
        c.append(get_result(10,result,gene3))
    print(np.mean(a,axis=0))
    print(np.mean(b,axis=0))
    print(np.mean(c,axis=0))

import pandas as pd
import tensorflow as tf
import sklearn
para_o = {
    'hidden_encoder_dim':20,                                    
    'hidden_decoder_dim':20, 
    'latent_dim':2,
    'lam':0,
    'epochs':600,
    'batch_size':10,
    'learning_rate':1e-4,
    'ran_walk':False,
    'check':False,
    'trade_off':0.5,
    'activation':tf.nn.relu,
    'optimizer':tf.train.AdamOptimizer,
    'norm':True,
    'decay':0.9,
    'initial':4,
    'PRE':sklearn.preprocessing.StandardScaler
        }
#para_o = pd.read_pickle('breastw.txt')
final('satimage',1,para_o)