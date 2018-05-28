# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:09:55 2018

@author: zhouying
"""
import numpy as np
import sys
sys.path.append('..\\')

def get_result_2(N,result,generation=[]):
	from myutil2 import app2,compute
	from sklearn.naive_bayes import GaussianNB
	if generation!=[]:
		gF_min = []
		ggmean = []
		gF_maj = []
		gacc = []
		gtp = []
		gauc = []
		for i in range(N):
			train,train_label,test,test_label = result[str(i)]
			gene = generation[str(i)]
			train,_ = app2(train,gene)
			train_label = np.concatenate((train_label,np.ones(gene.shape[0])),axis=0)
			gnb = GaussianNB()
			y_predne = gnb.fit(train,train_label).predict(test)
			y_pro = gnb.predict_proba(test)[:,1]
			re = compute(test_label,y_predne,y_pro)
		#    print('F1',temf,'AUC',tema,'gmean',temg)    
			gF_min.append(re[0])
			gF_maj.append(re[1])
			gacc.append(re[2])
			ggmean.append(re[3])
			gtp.append(re[4])
			gauc.append(re[5])
#		print('=====NB+vae=====')
#		print('mean F1-min:%.3f'%np.mean(gF_min),'mean f-maj:%.3f'%np.mean(gF_maj),'mean accuracy:%.3f'%np.mean(gacc))
#		print('mean gmean:%.3f'%np.mean(ggmean),'mean TPrate:%.3f'%np.mean(gtp),'mean AUC:%.3f'%np.mean(gauc))
	b = [np.mean(gF_min),np.mean(gF_maj),np.mean(gacc),np.mean(ggmean),np.mean(gtp),np.mean(gauc)]
	return b

def gene(positive):
    n,m = positive.shape
    r = np.random.normal(size=[n,m])
    sigma = np.std(positive,axis=0)/np.sqrt(n)
    for i in range(m):
        r[:,i]*=sigma[i]
    return positive-r

def normal(positive,size):
    ge1 = gene(positive)
    ge2 = gene(positive)
    ge2 = np.concatenate((ge1,ge2),axis=0)
    ge3 = gene(positive)
    ge3 = np.concatenate((ge2,ge3),axis=0)
    return ge1,ge2,ge3

def smote(positive):
    from myutil2 import Smote
    s =Smote(samples=positive)
    ge1 = s.over_sampling(n=1)
    s =Smote(samples=positive)
    ge2 = s.over_sampling(n=2)
    s =Smote(samples=positive)
    ge3 = s.over_sampling(n=3)
    return ge1,ge2,ge3