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


def onehot(labels):
    ''' one-hot 编码 '''
    n_sample = len(labels)
    n_class = max(labels) + 1
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
    def __init__(self,images,dtype='float'):
        self._images = images
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
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end,:]
    
def classify(nerror,perror,the=1):
    b_ne = (nerror **2)
    b_po = (perror **2)
    b_ne_1 = b_ne.sum(axis=1)
    b_po_1 = b_po.sum(axis=1)
    y_pre = []    
    for i in range(b_ne.shape[0]):
        if b_ne_1[i]< the*b_po_1[i]:
            y_pre.append(0)
        else:
            y_pre.append(1)
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
