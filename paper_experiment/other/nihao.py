# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:48:28 2018

@author: zhouying
"""

def random_pick(some_list, probabilities):  
    import random
    x = random.uniform(0,1)  
    cumulative_probability = 0.0  
    for item, item_probability in zip(some_list, probabilities):  
        cumulative_probability += item_probability  
        if x < cumulative_probability:break  
    return item  
  
def get_dataset(data,label,U=10):
    import numpy as np
    import sklearn
    h = []
    result = {}
    hh = []
    from generalized_ir import general_ir,weighted_general_ir,boundary_gir,consin_generalized_ir
    for i in range(U):
        deltaT = 1
        while deltaT >0:
            negative = data[label==0]
            index = [i for i,x in enumerate(label) if x==0]
            Tplas,Tminas,total = general_ir([data,label])
            total /=np.sum(total)
            deltaT = Tminas-Tplas
            reduce = random_pick(index,total[label==0])
            h.append(reduce)
            data = np.delete(data,reduce,0)
            label = np.delete(label,reduce,0)
        result[str(i)]={'data':data,'label':label}
    return 
    

def classifier(traindata,test,L=4):
    import sklearn
    import numpy as np
    for key in traindata.keys():
        data,label = traindata[key]
        cla = {}
        dx = np.ones(label.shape[0])/label.shape[0]
        for i in range(L):
            tem = sklearn.tree.DecisionTreeClassifier()
            traindata,trainlabel = sampling([data,label])
            tem.fit(traindata,trainlabel)
            cla[key+str(i)]=tem
            pre = tem.predict(data)
            error = 1-sklearn.metrics.accuracy_score(label,pre)
            if error>0.5:
                break
            belta = error/(1-error)
            dx[pre==label]*=belta
            dx /=sum(dx)            
    return cla

def classify(train,trainlabel,test,U=10,L=4):
    traindata = get_dataset(train,trainlabel,U)
    pre = classifier(traindata,L,test)            
    return pre

def evaluete(true,pred,pred_o):
    import myutil2
    ans = myutil2.compute(true,pred,pred_o)
    return ans[0],ans[5],ans[3]

def runH(name,fold):
    import numpy as np
    import scipy.io as scio
    mydata = scio.loadmat('MNIST_data\\UCI\\'+name+'.mat')
    data = np.array(mydata['data'])
    label = np.squeeze(mydata['label'])
    F1 = AUC = gmean = 0
    from myutil2 import create_cross_validation
    result = create_cross_validation([data,label],1,fold)
    for key in result.keys:
        train,trainlabel,test,testlabel = result[key]
        pre = classify(train,trainlabel,test)
        a,b,c = evaluete(trainlabel,pre)
        F1+=a/fold
        AUC +=b/fold
        gmean +=c/fold
    return F1,gmean,AUC
    
    
    
        
