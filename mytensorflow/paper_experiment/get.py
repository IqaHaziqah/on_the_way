def get_result(N,result,generation=[],name='None',write=False):
	from myutil2 import app2,compute
	
	import numpy as np
	import pandas as pd
	from sklearn.tree import DecisionTreeClassifier
	gnb = DecisionTreeClassifier(criterion='entropy')
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
		print('=====C4.5+vae=====')
		print('mean F1-min:%.3f'%np.mean(gF_min),'mean f-maj:%.3f'%np.mean(gF_maj),'mean accuracy:%.3f'%np.mean(gacc))
		print('mean gmean:%.3f'%np.mean(ggmean),'mean TPrate:%.3f'%np.mean(gtp),'mean AUC:%.3f'%np.mean(gauc))
		
	gF_min = []
	ggmean = []
	gF_maj = []
	gacc = []
	gtp = []
	gauc = []
	for i in range(N):
		train,train_label,test,test_label = result[str(i)]
		gnb = DecisionTreeClassifier(criterion='entropy')
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
	print('=====C4.5=====')
	print('mean F1-min:%.3f'%np.mean(gF_min),'mean f-maj:%.3f'%np.mean(gF_maj),'mean accuracy:%.3f'%np.mean(gacc))
	print('mean gmean:%.3f'%np.mean(ggmean),'mean TPrate:%.3f'%np.mean(gtp),'mean AUC:%.3f'%np.mean(gauc))
	a = [np.mean(gF_min),np.mean(gF_maj),np.mean(gacc),np.mean(ggmean),np.mean(gtp),np.mean(gauc)]
#def get_resultNB(N,result,generation,name='None',write=False):
#	import numpy as np
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
		print('=====NB+vae=====')
		print('mean F1-min:%.3f'%np.mean(gF_min),'mean f-maj:%.3f'%np.mean(gF_maj),'mean accuracy:%.3f'%np.mean(gacc))
		print('mean gmean:%.3f'%np.mean(ggmean),'mean TPrate:%.3f'%np.mean(gtp),'mean AUC:%.3f'%np.mean(gauc))
		
	gF_min = []
	ggmean = []
	gF_maj = []
	gacc = []
	gtp = []
	gauc = [] 
	for i in range(N):
		train,train_label,test,test_label = result[str(i)]
			
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
	print('=====NB=====')
	print('mean F1-min:%.3f'%np.mean(gF_min),'mean f-maj:%.3f'%np.mean(gF_maj),'mean accuracy:%.3f'%np.mean(gacc))
	print('mean gmean:%.3f'%np.mean(ggmean),'mean TPrate:%.3f'%np.mean(gtp),'mean AUC:%.3f'%np.mean(gauc))
	b = [np.mean(gF_min),np.mean(gF_maj),np.mean(gacc),np.mean(ggmean),np.mean(gtp),np.mean(gauc)]
#def get_resultNN(N,result,generation,name='None',write=False):
#	import numpy as np	
#	from myutil2 import app2,compute
	from sklearn.neighbors import KNeighborsClassifier
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
			
			
			gnb = KNeighborsClassifier(n_neighbors=3)
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
		print('=====3-NN+vae=====')
		print('mean F1-min:%.3f'%np.mean(gF_min),'mean f-maj:%.3f'%np.mean(gF_maj),'mean accuracy:%.3f'%np.mean(gacc))
		print('mean gmean:%.3f'%np.mean(ggmean),'mean TPrate:%.3f'%np.mean(gtp),'mean AUC:%.3f'%np.mean(gauc))
		c = [np.mean(gF_min),np.mean(gF_maj),np.mean(gacc),np.mean(ggmean),np.mean(gtp),np.mean(gauc)]
		n = int(train[train_label==1].shape[0]/gene.shape[0]) 
	import csv
	if write == True:
		with open('test.csv','a') as f:
			writer = csv.writer(f)
			writer.writerow(name+str(n))
			writer.writerows([a,b,c])                                                                                                                          
			#	gF_min = []
	ggmean = []
	gF_maj = []
	gacc = []
	gtp = []
	gauc = []
	for i in range(N):
		train,train_label,test,test_label = result[str(i)]
		
		
		gnb = KNeighborsClassifier(n_neighbors=3)
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
	print('=====3-NN=====')
	print('mean F1-min:%.3f'%np.mean(gF_min),'mean f-maj:%.3f'%np.mean(gF_maj),'mean accuracy:%.3f'%np.mean(gacc))
	print('mean gmean:%.3f'%np.mean(ggmean),'mean TPrate:%.3f'%np.mean(gtp),'mean AUC:%.3f'%np.mean(gauc))
	c = [np.mean(gF_min),np.mean(gF_maj),np.mean(gacc),np.mean(ggmean),np.mean(gtp),np.mean(gauc)]
	return a,b,c