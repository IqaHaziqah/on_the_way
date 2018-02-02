# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:48:18 2018

@author: zhouying
"""

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
               'breasttissue','movement','spect','vertebral','breastw']
    from generalized_ir import generalized_ir,weighted_generalized_ir
    import numpy as np
    from pandas import DataFrame
    from myutil2 import create_cross_validation
    from overlapping import overlapping
    import scipy.io as scio
    import scipy.stats as st
    from get import get_result
    column = ['samples','attributes','minority','majority','imbalance-ratio',
              'general-ir','wei-gir','overlapping','C4.5Fmeasure','C4.5gmean',
              'C4.5AUC',
              'NBFmeasure','NBgmean','NBAUC','3nnFmeasure','3nngmean','3nnAUC']
    record = []
    for value in dataset:
        mydata = scio.loadmat('MNIST_data\\UCI\\'+value+'.mat')
        data = mydata['data']
        label = mydata['label']
        if label.shape[0]==1:
            label = label.squeeze(0)
        elif label.shape[1]==1:
            label = label.squeeze(1)  
        myresult = create_cross_validation([data,label],1,10)
        print(value)
        ov = overlapping([data,label])
        gir = generalized_ir([data,label])
        wgir = weighted_generalized_ir([data,label])
        a,b,c = get_result(10,myresult)
        samples = data.shape[0]
        attributes = data.shape[1]
        minority = np.count_nonzero(label)
        majority = samples-minority
        ir = float(majority)/minority
        record.append([samples,attributes,minority,majority,ir,gir,wgir,ov,a[0],a[3],a[5],b[0],b[3],b[5],c[0],c[3],c[5]])
        
    record = DataFrame(record,columns=column,index=dataset)
    for value in ['samples','imbalance-ratio','general-ir','wei-gir','overlapping']:
        for j in ['C4.5Fmeasure','C4.5gmean','C4.5AUC','NBFmeasure','NBgmean','NBAUC','3nnFmeasure','3nngmean','3nnAUC']:
            print(value,j)
            print(st.pearsonr(record[value],record[j]))
#    record.to_csv('tet.csv',columns=column,index=dataset)