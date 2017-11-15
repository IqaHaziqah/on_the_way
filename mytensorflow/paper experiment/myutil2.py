#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:45:40 2017

@author: zhouying
"""

import tensorflow as tf
import numpy as np

def xavier_init(arg,constant = 1):
    fan_in, fan_out = arg
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                            minval = low, maxval = high,
                            dtype = tf.float32)

def onehot(labels,n_class):
    ''' one-hot 编码 '''
    n_sample = len(labels)    
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1

    return onehot_labels


def crossvalidation(data,divide_rate):
    from sklearn.cross_validation import StratifiedKFold
    dic = []
    skf = StratifiedKFold(data,n_folds=int(1/divide_rate))
#    i = 1
    for train,test in skf:
        dic.append([train,test])        
    return dic

def randomselect(data,batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

class Dataset(object):
    def __init__(self,images,label=None,dtype='float'):
        if label == None:
            self._images = images
            self.xx = self._images.shape[1]
            self.flag = None
        else:
            if images.shape[0]==label.shape[0]:
                self._images = np.concatenate((images,label),axis=1)
            else:
                self._images = np.concatenate((images,label.T),axis=1)
            self.xx = self._images.shape[1]-1
            self.flag = True
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = images.shape[0]
        
    def images(self):
        return self.images
    
    def num_examples(self):
        return self._num_examples
    
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch >self._num_examples:
            self._epochs_completed +=1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            tem = np.zeros([self._num_examples,self._images.shape[1]])
            for i in range(len(perm)):
                tem[i,:] = self._images[perm[i],:]
            self._images = tem
            start = 0
            self._index_in_epoch = batch_size
#            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        
#        print(xx)
        if self.flag:
            l = np.array(self._images[start:end,self.xx-1]).reshape(1,batch_size)
        else:
            l = []
        return [self._images[start:end,0:self.xx],l]
    
def classify(gama,nerror,perror,label=[0,1]):
    b_ne = (nerror **2)
    b_po = (perror **2)
    b_ne_1 = b_ne.sum(axis=1)
    b_po_1 = b_po.sum(axis=1)
    b_ne_2 = np.exp(-gama*b_ne_1)
    b_po_2 = b_po_1
    y_pre = []    
    for i in range(b_ne.shape[0]):
        if b_ne_2[i]< b_po_2[i]:
            y_pre.append(label[0])
        else:
            y_pre.append(label[1])
    return y_pre
  
import random
from sklearn.neighbors import NearestNeighbors
class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
#        print('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1

def random_walk(z,gene_size):
    import numpy as np
    np.random.seed(0)
    z = np.array(z)
    if len(z.shape)>2:
        z = z[0]
    sigma = np.power(np.var(z,axis=0)/z.shape[0],0.5)
    z_sample = []
    k = 0
    j = 0
    while(k<gene_size):
        z_gene = z[j]-sigma*np.random.normal(0,1,z.shape[1])
#        z_gene = z[j]-sigma*np.random.normal(0,1)
        z_sample.append(z_gene)
        k = k+1
        j = k%z.shape[0]
        if j == 0:
            np.random.shuffle(z)
    return np.array(z_sample)

def app(positive,negative,gene=[]):
    positive = np.row_stack([positive,gene])
    a = np.ones(positive.shape[0])
    b = np.zeros(negative.shape[0])        
    return np.row_stack((positive,negative)),np.append(a,b)

# need more knowledge
def rescon(positive,negative,x_train,x_test,feed_dict):
    from SDAE import mysdae
    from myutil import classify
    epoch = feed_dict['training_epochs']
    hidden_size = feed_dict['hidden_size']
    keep_rate = feed_dict['keep_rate']
    scale = feed_dict['scale']
    the = feed_dict['the']
    hidden_size_positive = feed_dict['hidden_size_positive']
    tr_ne,b_ne = mysdae(negative,epoch,(x_train,x_test),stack_size = len(hidden_size),
                        hidden_size=hidden_size,keep_rate=keep_rate,scale=scale)
    tr_po,ge_po = mysdae(positive,epoch,(x_train,x_test),stack_size = len(hidden_size),
                        hidden_size=hidden_size_positive,keep_rate=keep_rate,scale=scale)
    tr_pre = classify(the,tr_ne,tr_po)
    y_pre = classify(the,b_ne,ge_po)
    return tr_pre,y_pre

def compute(y_true,y_pre):
    from sklearn import metrics
    F1 = metrics.f1_score(y_true, y_pre)
    a = metrics.confusion_matrix(y_true, y_pre)
    b = (a[0][0]/(a[0][0]+a[0][1]))*(a[1][1]/(a[1][0]+a[1][1]))
    g_mean = pow(b,0.5)
    auc = metrics.roc_auc_score(y_true, y_pre)
    return F1,g_mean,auc

def write(path,para,dc):    
    from xlrd import open_workbook
    from xlutils.copy import copy
    import numpy as np
    
    rexcel = open_workbook(path) # 用wlrd提供的方法读取一个excel文件
    rows = rexcel.sheets()[0].nrows # 用wlrd提供的方法获得现在已有的行数
    excel = copy(rexcel) # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
    table = excel.get_sheet(0) # 用xlwt对象的方法获得要操作的sheet
    row = rows+1    
    j = 0
    for index,value in para.items():
        table.write(row,j+1,index)
        table.write(row,j+2,value)
        j +=2    
    row +=1    
    for key in dc.keys():
        table.write(row, 0, key) # xlwt对象的写方法，参数分别是行、列、值
        tmp = dc[key]        
        for j,value in enumerate(tmp):
            table.write(row, j+1, (value))        
        table.write(row,j+2,(np.mean(tmp)))
        row += 1
    excel.save(path) # xlwt对象的保存方法，这时便覆盖掉了原来的excel
    return

def cross_validation(data,label,para_c,para_o):
    kfold = para_c['kfold']
    neg = 0
    pos = 1
    gF1 = []
    ggmean = []
    gauc = []
    path = 'collection.xls'
    from vae4 import mnist_vae
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import MinMaxScaler
    skf = StratifiedKFold(n_splits = kfold)
    for train_index,test_index in skf.split(data,label):
        train = data[train_index]
        test = data[test_index]
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit_transform(train)
        min_max_scaler.transform(test)
        train_label = label[train_index]
        test_label = label[test_index]
        negative = train[train_label==neg]
        positive = train[train_label==pos]
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        if para_c['over_sampling'] =='SMOTE':
            s = Smote(positive,N=100)
            gene = s.over_sampling()
#            print('1')
        elif para_c['over_sampling'] == 'vae':
            gene_size = negative.shape[0]-positive.shape[0]
            gene = mnist_vae(positive,gene_size,para_o)
#            print('2')
        elif para_c['over_sampling'] == 'random_walk':
            gene_size = positive.shape[0]
            gene = random_walk(positive,gene_size)
#            print('3')
#            print(gene)
        else:
            gene=[]
#            print('4')
#            print('else gene')
        train,train_label = app(positive,negative,gene)
#        print(train.shape)
        y_predne = gnb.fit(train,train_label).predict(test)
        temf,temg,tema = compute(test_label,y_predne)
        gF1.append(temf)
        ggmean.append(temg)
        gauc.append(tema)
    print('##########################zhouying###################################')
    if para_c['over_sampling'] == 'vae':
        write(path,dict(para_c,**para_o),{'F1':gF1,'AUC':gauc,'gmean':ggmean})
    else:
        write(path,para_c,{'F1':gF1,'AUC':gauc,'gmean':ggmean})
    return    