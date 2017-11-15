#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:51:18 2017

@author: zhouying
"""
#from __future__ import division, print_function, absolute_import

#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import preprocessing
import vae
#%matplotlib inline
from SDAE import mysdae
from myutil import classify,Smote
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#ionosphere yeast glass
data=np.loadtxt('./MNIST_data/ionosphere.txt',dtype='float32')

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 10

divide_rate = 0.3
keep_rate = 0.5
# Network Parameters
#n_hidden_1 = int(data.shape[1]*0.7)+1 # 1st layer num features
#n_hidden_2 = 128 # 2nd layer num features
n_input = data.shape[1]-1 # MNIST data input (img shape: 28*28)
hidden_size = [25]
hidden_size_positive = [25]
scale = 0
# tf Graph input (only pictures)

my_model_F1 = []
my_model_auc = []
my_F1 = []
my_auc = []
my_model_gene = []
my_model_gen_auc = []
my_model_gen_f1 = []
my_model_gen_gmean = []
F1 = []
my_smote = []

auc = []
model_gene = []
from sklearn.cross_validation import StratifiedKFold
skf = StratifiedKFold(data[:,n_input],n_folds=int(1/divide_rate))
for train_1,test_1 in skf:
    train = data[train_1,:]
    test = data[test_1,:]
    negative = []
    positive = []#np.empty([None,9])
    
    
#     preprocessing ,normalize it into (0,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    train = np.column_stack((min_max_scaler.fit_transform(train[:,0:n_input]),train[:,n_input]))
    test = np.column_stack((min_max_scaler.transform(test[:,0:n_input]),test[:,n_input]))
    # end of preprocessing
    
    #divide into negative data and positive ones
    for i in range(train.shape[0]):
        if train[i,n_input] == 0:
            negative.append(train[i,:])
        else:
            positive.append(train[i,:])
    negative = np.array(negative)
    positive = np.array(positive)
    x_train_negative = negative[:,0:n_input]
    x_train_positive = positive[:,0:n_input]
    #end of divide
    x_train_orginal = x_train_positive
    
#    genetate the x_train_positive to be balance dataset
    y_train_positive = np.ones(x_train_positive.shape[0])
    if x_train_positive.shape[0] < x_train_negative.shape[0]/2:
        gene_size = x_train_positive.shape[0]
    else:
        gene_size = x_train_negative.shape[0]-x_train_positive.shape[0]
    gene = vae.myvae(x_train_positive,y_train_positive,gene_size)
    gene = np.reshape(gene,[len(gene),x_train_positive.shape[1]])
#    gene = min_max_scaler.transform(gene)

    gene = min_max_scaler.fit_transform(gene)
    x_train_positive = np.row_stack((x_train_positive,gene))
    
    s = Smote(x_train_orginal,100)
    gene_2 = s.over_sampling()
    x_train_positive_2 = np.row_stack((x_train_orginal,gene_2))
    # end of generation
#in the process, we didn't change the original dataset such as train and test except preprocessing   
    x_train = train[:,0:n_input]
    y_train = train[:,n_input]
    x_test = test[:,0:n_input]
    y_test = test[:,n_input]


#the negative reconstruction model
    
#    b_ne = mysdae(x_test)
#    tr_ne = mysdae(x_train)
    tr_ne,b_ne = mysdae(x_train_negative,(x_train,x_test),stack_size = len(hidden_size),
                        hidden_size=hidden_size,keep_rate=keep_rate,scale=scale)
    print("Sess1 negative model Optimization Finished!")
# the positive reconstruction model 
    
#    ge_po = mysdae(x_test)
#    tr_po = sess2.run(reconstruction_error(x_train)) 
    tr_po,ge_po = mysdae(x_train_positive,(x_train,x_test),stack_size = len(hidden_size_positive),
                        hidden_size=hidden_size_positive,keep_rate=keep_rate,scale=scale)
    print("Sess2 generation positive model Optimization Finished!")

#the positive without generation model
     
    tr_po_2,b_po = mysdae(x_train_orginal,(x_train,x_test),stack_size = len(hidden_size_positive),
                        hidden_size=hidden_size_positive,keep_rate=keep_rate,scale=scale)
    print("Sess3 original positive model Optimization Finished!")

    tr_po_3,ge_po_2 = mysdae(x_train_positive_2,(x_train,x_test),stack_size = len(hidden_size_positive),
                        hidden_size=hidden_size_positive,keep_rate=keep_rate,scale=scale)
    print("Sess4 generation smoted positive model Optimization Finished!")
#count the recontrustion error and then compare in the two model

    
#predict the labels
    y_pre = classify(b_ne,b_po)
    y_newpre = classify(b_ne,ge_po)        
    tr_pre = classify(tr_ne,tr_po)
    tr_pre_2 = classify(tr_ne,tr_po_2)
    y_s = classify(b_ne,ge_po_2)
    

##################################
#compute F1 score
    from sklearn import metrics
    my_model_F1.append(metrics.f1_score(y_test, y_pre))
    my_model_auc.append(metrics.f1_score(y_train, tr_pre_2))
    my_F1.append(metrics.f1_score(y_train,tr_pre))
    my_auc.append(metrics.roc_auc_score(y_train, tr_pre))
    my_model_gene.append(metrics.f1_score(y_test, y_newpre))
    my_model_gen_auc.append(metrics.roc_auc_score(y_test, y_newpre))
    my_smote.append(metrics.f1_score(y_test,y_s))
    
#    my_cc.append(metrics.confusion_matrix(y_test, y_pre))
#    my_model_gmean.append(metrics.gmean_score(y_test, y_pre))
    a = metrics.confusion_matrix(y_test, y_newpre)
    b = (a[0][0]/(a[0][0]+a[0][1]))*(a[1][1]/(a[1][0]+a[1][1]))
    g_mean = pow(b,0.5)
    my_model_gen_gmean.append(g_mean)
    print('my model test F1:',metrics.f1_score(y_test, y_pre))
    print('my model train f1:',metrics.f1_score(y_train, tr_pre_2))
    print('my model generation test F1:',metrics.f1_score(y_test, y_newpre))
#    print('my model generation g-mean:',g_mean)
#    print('my model generation auc:',metrics.roc_auc_score(y_test, y_newpre))
#    print('my model:',metrics.confusion_matrix(y_test, y_newpre))
    
    


####################################
#compute the normal classification's F1 score

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(train[:,0:n_input], train[:,n_input]).predict(x_test)
    F1.append(metrics.f1_score(y_test, y_pred))
    auc.append(metrics.roc_auc_score(y_test, y_pred))
    
#    cc.append(metrics.roc_auc_score(y_test, y_pred))
    print('NB test F1:',metrics.f1_score(y_test, y_pred))
#    print('roc:',metrics.roc_auc_score(y_test, y_pred))
    
#    print('GBN:',metrics.confusion_matrix(y_test, y_pred))
    

#predict the generated samples
    gnb = GaussianNB()
    a = np.ones(x_train_positive_2.shape[0])
    b = np.zeros(x_train_negative.shape[0])
    y_predne = gnb.fit(np.row_stack((x_train_positive_2,x_train_negative)),np.append(a,b)).predict(x_test)
    print('NB generation test F1:',metrics.f1_score(y_test, y_predne))
    model_gene.append(metrics.f1_score(y_test, y_predne))

print('#######################zhouying########################')
print('my_model_mean_test_F1',np.mean(my_model_F1))
print('my_model_mean_train_f1',np.mean(my_model_auc))
print('my_mean_F1',np.mean(my_F1))
print('my_mean_auc',np.mean(my_auc))
print('my_mean_smote_f1',np.mean(my_smote))
#print('my_mean_cc',np.mean(my_cc))
print('my_mean_generation_F1',np.mean(my_model_gene))
print('my_mean_generation_auc',np.mean(my_model_gen_auc))
print('my_mean_generation_gmean',np.mean(my_model_gen_gmean))
#print('my_mean_generation_bagging_f1',np.mean(my_model_gen_f1))
print('GNB_mean_F1',np.mean(F1))
print('GNB_mean_auc',np.mean(auc))
print('GNB_mean_generation_F1',np.mean(model_gene))
#print('mean_cc',np.mean(cc))