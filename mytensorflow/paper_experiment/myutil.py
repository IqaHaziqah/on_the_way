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
    z = np.array(z)[0]
    
    sigma = np.var(z,axis=0)/np.power(z.shape[0],0.5)
#    ran = np.random.normal(0,1,z.shape[1])
#    print(sigma.shape)
#    random = np.random.normal(0,1,gene_size)
    z_sample = []
    k = 0
    j = 0
    while(k<gene_size):
#        print(z[j].shape)
#        print((sigma*np.random.normal(0,1,z.shape[1])).shape)
        z_gene = z[j]-sigma*np.random.normal(0,10,z.shape[1])
#        print(z_gene.shape)
        z_sample.append(z_gene)
        k = k+1
        j = k%z.shape[0]
        if k%z.shape[0] == 0:
            np.random.shuffle(z)
    return np.array(z_sample)

def app(positive,negative,gene=[]):
    positive = np.row_stack([positive,gene])
    a = np.ones(positive.shape[0])
    b = np.zeros(negative.shape[0])        
    return np.row_stack((positive,negative)),np.append(a,b)

# need more knowledge
def rescon(positive,negative,x_train,x_test,**feed_dict):
    from SDAE import mysdae
    from myutil import classify
    tr_ne,b_ne = mysdae(negative,epoch,(x_train,x_test),stack_size = len(hidden_size),
                        hidden_size=hidden_size,keep_rate=keep_rate,scale=scale)
    tr_po,ge_po = mysdae(positive,epoch,(x_train,x_test),stack_size = len(hidden_size),
                        hidden_size=hidden_size_positive,keep_rate=keep_rate,scale=scale)
    tr_pre = classify(the,tr_ne,tr_po)
    y_pre = classify(the,b_ne,ge_po)
    return tr_pre,y_pre

def compute(y_pre,y_true):
    from sklearn import metrics
    F1 = metrics.f1_score(y_true, y_pre)
    a = metrics.confusion_matrix(y_true, y_pre)
    b = (a[0][0]/(a[0][0]+a[0][1]))*(a[1][1]/(a[1][0]+a[1][1]))
    g_mean = pow(b,0.5)
    auc = metrics.roc_auc_score(y_true, y_pre)
    return F1,g_mean,auc

import xlwt;
import xlrd;
#import xlutils;
from xlutils.copy import copy;
 
styleBoldRed   = xlwt.easyxf('font: color-index red, bold on');
headerStyle = styleBoldRed;
wb = xlwt.Workbook();
ws = wb.add_sheet(gConst['xls']['sheetName']);
ws.write(0, 0, "Header",        headerStyle);
ws.write(0, 1, "CatalogNumber", headerStyle);
ws.write(0, 2, "PartNumber",    headerStyle);
wb.save(gConst['xls']['fileName']);
 
#open existed xls file
#newWb = xlutils.copy(gConst['xls']['fileName']);
#newWb = copy(gConst['xls']['fileName']);
oldWb = xlrd.open_workbook(gConst['xls']['fileName'], formatting_info=True);
print(oldWb); #<xlrd.book.Book object at 0x000000000315C940>
newWb = copy(oldWb);
print(newWb); #<xlwt.Workbook.Workbook object at 0x000000000315F470>
newWs = newWb.get_sheet(0);
newWs.write(1, 0, "value1");
newWs.write(1, 1, "value2");
newWs.write(1, 2, "value3");
print("write new values ok");
newWb.save(gConst['xls']['fileName']);
print("save with same name ok");