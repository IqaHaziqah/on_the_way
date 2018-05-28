# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:40:12 2018

@author: zhouying
"""
import sys
sys.path.append('..\\')
from ndo import normal,smote
def ten_fold(name,para_o=[],method='normal'):
    import numpy as np
    import scipy.io
    filepath = '..\\MNIST_data\\UCI\\'+name+'.mat'
    mydata = scipy.io.loadmat(filepath)
    data = np.array(mydata['data'])
    label = np.transpose(mydata['label'])
    label = np.squeeze(label)
   
    from myutil2 import create_cross_validation
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
        if method =='normal':
            gene_1,gene_2,gene_3 = normal(togene,togene.shape[0])
        else:
            gene_1,gene_2,gene_3 = smote(togene)
        generation_1[str(i)] = gene_1
        generation_2[str(i)] = gene_2
        generation_3[str(i)] = gene_3    
    
    return result,generation_1,generation_2,generation_3  #K折交叉验证 每次跑K回

def final(name,fold,method,para_o=[]):    
    import numpy as np
    a = []
    b = []
    c = []
    from ndo import get_result_2
    for i in range(fold):    
        result,gene1,gene2,gene3 = ten_fold(name,para_o,method)
        a.append(get_result_2(10,result,gene1))
        b.append(get_result_2(10,result,gene2))
        c.append(get_result_2(10,result,gene3))
    print(np.mean(a,axis=0))
    print(np.mean(b,axis=0))
    print(np.mean(c,axis=0))
import sklearn
para_o = {}
para_o['PRE']= sklearn.preprocessing.StandardScaler
#para_o = pd.read_pickle('breastw.txt')
#ans = ten_fold(name='ionosphere',para_o=para_o)
dataset = ['breastw','vehicle','segment-challenge','diabetes','ionosphere','sonar']
for name in ['german']:
    print('smote')
    print(name)
    final(name,10,'smote',para_o)
    print('normal')
    print(name)
    final(name,10,'normal',para_o)
