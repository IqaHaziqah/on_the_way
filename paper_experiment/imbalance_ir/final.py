# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:48:18 2018

@author: zhouying
"""
import sys
sys.path.append('F:\\OneDrive\\mytensorflow\\paper_experiment')
def final():
    dataset = ['diabetes','ionosphere','satimage','segmentchallenge','vehicle','sonar']

    #dataset = ['vehicle']
    import os,pickle
    from get import get_result
    
    for value in dataset:
        filedir = value
        if os.path.exists(value) and os.path.isfile(filedir+'\\result'):             
            with open(filedir+'.\\result','rb') as f:
                result = pickle.load(f)
            f.close()
            with open(filedir+'.\\generation1','rb') as f:
                generation = pickle.load(f)
            f.close()     
            print('###@@@==========='+value+'============@@@###')
            print('###@@@=====100%=====@@@###')
            get_result(10,result,generation,value,False)
            with open(filedir+'.\\result','rb') as f:
                result = pickle.load(f)
            f.close()
            with open(filedir+'.\\generation2','rb') as f:
                generation = pickle.load(f)
            f.close()     
    #        print('###@@@======'+value+'======@@@###')
            print('###@@@=====200%=====@@@###')
            get_result(10,result,generation,value,False)
            with open(filedir+'.\\result','rb') as f:
                result = pickle.load(f)
            f.close()
            with open(filedir+'.\\generation3','rb') as f:
                generation = pickle.load(f)
            f.close()     
    #        print('###@@@======'+value+'======@@@###')
            print('###@@@=====300%=====@@@###')
            get_result(10,result,generation,value,False)
        else:
            print(filedir) 
            continue
    return


def compare_imbalance():
    dataset = ['diabetes','ionosphere','satimage','segment-challenge','vehicle',
               'sonar','german','wpbc','glass','haberman',
               'breasttissue','movement','spect','vertebral','breastw',
               'yeast1','yeast2','yeast0','yeast6',]
#               'abalone6',
#               'abalone7','abalone8','abalone9','abalone10','abalone11','abalone12']
    from generalized_ir import generalized_ir,weighted_generalized_ir,weighted_consin_generalized_ir,weighted_gaussian_generalized_ir,weighted_normalize_generalized_ir
    import numpy as np
    from pandas import DataFrame
    import pandas as pd
    from myutil2 import create_cross_validation
    from overlapping import overlapping,weight_overlapping,another_overlapping
    import scipy.io as scio
    import scipy.stats as st
    from get import get_result_2
    from measurement import compute_cm_fixedk,compute_f1
    import os
    column = ['samples','attributes','minority','majority','imbalance-ratio',
              'gir','igir','wei-gir','wei-igir','cm','f1']
    ''''wngir',
              'overlapping','aov','wei-ov',
              'wei-girov','wei-giraov','''
    record = []
    for value in dataset:
        mydata = scio.loadmat('..\\MNIST_data\\UCI\\'+value+'.mat')
        data = mydata['data']
        label = mydata['label']
        if label.shape[0]==1:
            label = label.squeeze(0)
        elif label.shape[1]==1:
            label = label.squeeze(1)  
#        print(value)
#        ov = overlapping([data,label])
#        wov = weight_overlapping([data,label])
#        aov = another_overlapping([data,label])
        gir = generalized_ir([data,label])
        wgir = weighted_generalized_ir([data,label])
        wcgir = weighted_consin_generalized_ir([data,label])
        wggir = weighted_gaussian_generalized_ir([data,label])
        cm = compute_cm_fixedk([data,label])
        f1 = compute_f1([data,label])
#        wngir = weighted_normalize_generalized_ir([data,label])
#        wgirov = 0.1*wov+wgir
#        wgiraov = 0.1*aov+wgir
#        a,b,c,d = ten_get_result([data,label])
        samples = data.shape[0]
        attributes = data.shape[1]
        minority = np.count_nonzero(label)
        majority = samples-minority
        ir = float(majority)/minority
        record.append([samples,attributes,minority,majority,ir,gir[0],gir[1],wgir[0],wgir[1],cm,f1])
    record = DataFrame(record,columns=column,index=dataset)
    if os.path.isfile('dataset.txt'):
        classfy = pd.read_csv('dataset.txt',index_col=0)
    record = pd.merge(record,classfy,how='left',left_index=True,right_index=True)
#    re = ['attributes','general-ir','wei-gir','wcgir','wggir']#,'wngir','overlapping','aov','wei-ov','wei-girov','wei-giraov'
#    row = ['C4.5Fmeasure','C4.5gmean','C4.5AUC','NBFmeasure','NBgmean','NBAUC','3nnFmeasure','3nngmean','3nnAUC','SGDFmeasure','SGDgmean','SGDAUC']
    row = ['C4.5Fmeasure','C4.5gmean','C4.5TP','C4.5AUC',
              'NBFmeasure','NBgmean','NBTP','NBAUC',
              '5nnFmeasure','5nngmean','5nnTP','5nnAUC',
              'SGDFmeasure','SGDgmean','SGDTP','SGDAUC']
    app = []
    result=[]
    for value in row:
        for j in column:
            app.append(st.pearsonr(record[value],record[j])[0])
        result.append(app)
        app= []
    result = DataFrame(result,columns=column,index=row)
    return result,record
            
#    record.to_csv('tet.csv',columns=column,index=dataset)

def ten_get_result(arg):
    '''10次10折交叉验证结果均值，返回少数类的F1值、gmean、AUC的平均值'''
    from myutil2 import create_cross_validation
    from get import get_result_2
    import numpy as np
    data,label = arg
    aa = []
    bb = []
    cc = []
    dd = []
    for i in range(10):
        myresult = create_cross_validation([data,label],1,10)
        a,b,c,d = get_result_2(10,myresult)
        aa.append(a)
        bb.append(b)
        cc.append(c)
        dd.append(d)
    aa = np.array(aa)
    bb = np.array(bb)
    cc = np.array(cc)
    dd = np.array(dd)
    return np.mean(aa,axis=0),np.mean(bb,axis=0),np.mean(cc,axis=0),np.mean(dd,axis=0)

def save():
    '''保存数据集的十次十折交叉验证的平均结果，并返回其F1值、gmean、sensitivity、AUC值等'''
    import scipy.io as scio
    import pandas as pd
    import os
    column = ['C4.5Fmeasure','C4.5gmean','C4.5TP','C4.5AUC',
              'NBFmeasure','NBgmean','NBTP','NBAUC',
              '5nnFmeasure','5nngmean','5nnTP','5nnAUC',
              'SGDFmeasure','SGDgmean','SGDTP','SGDAUC']
    dataset = ['diabetes','ionosphere','satimage','segment-challenge','vehicle',
           'sonar','german','wpbc','glass','haberman',
           'breasttissue','movement','spect','vertebral','breastw',
           'yeast1','yeast2','yeast0','yeast6','abalone6',
           'abalone7','abalone8','abalone9','abalone10','abalone11','abalone12']
    record = []
    for value in dataset:
        print(value)
        mydata = scio.loadmat('..\\MNIST_data\\UCI\\'+value+'.mat')
        data = mydata['data']
        label = mydata['label']
        if label.shape[0]==1:
            label = label.squeeze(0)
        elif label.shape[1]==1:
            label = label.squeeze(1)
        a,b,c,d = ten_get_result([data,label])
        record.append([a[0],a[3],a[4],a[5],b[0],b[3],b[4],b[5],c[0],c[3],c[4],c[5],d[0],d[3],d[4],d[5]])
    record = pd.DataFrame(record,columns=column,index=dataset)
    if os.path.isfile('dataset.txt'):
        os.remove('dataset.txt')
    record.to_csv('dataset.txt',columns=column,index=dataset)
    return 
    
#ans = compare_imbalance()

import pandas as pd
ans = pd.read_pickle('ans.txt')
#pd.to_pickle(ans,'ans.txt')

da = ['diabetes','ionosphere','satimage',"segment-challenge",'vehicle',
               'sonar','german','wpbc','glass','haberman',
               'breasttissue','movement','spect','vertebral','breastw',
               'yeast1','yeast2','yeast0','yeast6',]
import matplotlib.pyplot as plt
evalua = ["imbalance-ratio",
              'gir','wei-igir','cm','f1']
M =8
N = 8

import scipy.stats as st
for j in ["f1"]:
#    plt.figure(figsize=(2.4,1.6))
    for i in da:
        plt.scatter(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],marker='o',s=5,)
#        if 'yeast' not in i and 'br' not in i:
#            plt.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],i[:2],fontsize=M)
#        elif  'br' in i:
#            plt.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'br'+i[-1],fontsize=M)
#        else:
#            plt.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'ye'+i[-1],fontsize=M)
    plt.xlabel('Sensitivity',fontsize=N)
    plt.ylabel("F1",fontsize=N)
#    plt.title("UCI Dataset",fontsize=N)
#    plt.ylim([0,1])
    plt.xlim([0,1])
    fig = plt.gcf()
#    plt.setp(fig,fontsize=10)
    fig.set_size_inches(w=3,h=3)
    fig.tight_layout()
    plt.savefig(j+'.png',dpi=300)
##        plt.ylim([0,1])

#plt.figure()
#ax = plt.subplot2grid((3,4),(0,0),colspan=2)
#for j in ["imbalance-ratio"]:
##    plt.figure(figsize=(2.4,1.6))
#    for i in da:
#        ax.scatter(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],marker='o')
#        if 'yeast' not in i and 'br' not in i:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],i[:2],fontsize=M)
#        elif  'br' in i:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'br'+i[-1],fontsize=M)
#        else:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'ye'+i[-1],fontsize=M)
#    ax.set_xlabel('Sensitivity',fontsize=N)
#    ax.set_ylabel("IR",fontsize=N)
##    plt.title("UCI Dataset",fontsize=N)
#    ax.set_xlim([0,1])
#ax = plt.subplot2grid((3,4),(0,2),colspan=2)
#for j in ["f1"]:
##    plt.figure(figsize=(2.4,1.6))
#    for i in da:
#        ax.scatter(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],marker='o')
#        if 'yeast' not in i and 'br' not in i:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],i[:2],fontsize=M)
#        elif  'br' in i:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'br'+i[-1],fontsize=M)
#        else:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'ye'+i[-1],fontsize=M)
#    ax.set_xlabel('Sensitivity',fontsize=N)
#    ax.set_ylabel("F1",fontsize=N)
##    plt.title("UCI Dataset",fontsize=N)
#    ax.set_xlim([0,1])
#ax = plt.subplot2grid((3,4),(1,0),colspan=2)
#for j in ["cm"]:
##    plt.figure(figsize=(2.4,1.6))
#    for i in da:
#        ax.scatter(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],marker='o')
#        if 'yeast' not in i and 'br' not in i:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],i[:2],fontsize=M)
#        elif  'br' in i:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'br'+i[-1],fontsize=M)
#        else:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'ye'+i[-1],fontsize=M)
#    ax.set_xlabel('Sensitivity',fontsize=N)
#    ax.set_ylabel("CM",fontsize=N)
##    plt.title("UCI Dataset",fontsize=N)
#    ax.set_xlim([0,1])
#ax = plt.subplot2grid((3,4),(1,2),colspan=2)
#for j in ["gir"]:
##    plt.figure(figsize=(2.4,1.6))
#    for i in da:
#        ax.scatter(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],marker='o')
#        if 'yeast' not in i and 'br' not in i:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],i[:2],fontsize=M)
#        elif  'br' in i:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'br'+i[-1],fontsize=M)
#        else:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'ye'+i[-1],fontsize=M)
#    ax.set_xlabel('Sensitivity',fontsize=N)
#    ax.set_ylabel("GIR",fontsize=N)
##    plt.title("UCI Dataset",fontsize=N)
#    ax.set_xlim([0,1])
#ax = plt.subplot2grid((3,4),(2,1),colspan=2)
#for j in ["igir"]:
##    plt.figure(figsize=(2.4,1.6))
#    for i in da:
#        ax.scatter(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],marker='o')
#        if 'yeast' not in i and 'br' not in i:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],i[:2],fontsize=M)
#        elif  'br' in i:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'br'+i[-1],fontsize=M)
#        else:
#            ax.text(ans[1].loc[i,'C4.5TP'],ans[1].loc[i,j],'ye'+i[-1],fontsize=M)
#    ax.set_xlabel('Sensitivity',fontsize=N)
#    ax.set_ylabel("IGIR",fontsize=N)
##    plt.title("UCI Dataset",fontsize=N)
#    ax.set_xlim([0,1])

#from pandas import DataFrame
#from myutil2 import computeCorrelation
#result = []
#
##calss = ["C4.5gmean","C4.5Fmeasure","C4.5AUC","C4.5TP"]
#calss = ['C4.5Fmeasure','C4.5gmean','C4.5TP','C4.5AUC',
#              'NBFmeasure','NBgmean','NBTP','NBAUC',
#              '5nnFmeasure','5nngmean','5nnTP','5nnAUC',
#              'SGDFmeasure','SGDgmean','SGDTP','SGDAUC']
#for j in evalua:
#    r_value = []
#    for i in calss:
#        x = ans[1].loc[:,j]
#        y = ans[1].loc[:,i]
#        r_value.append(computeCorrelation(x,y))
#    result.append(r_value)
##        print(r_value)
#result = DataFrame(result,index=evalua,columns=calss)    
#    

    