def mnist_vae(data,gene_size,feed_dict):
    import numpy as np
    import pandas as pd
    from vae4 import mnist_vae
    seper = []
    for i in range(data.shape[1]):
    	seper.append(len(set(data[:,i])))
    mask = np.array(seper)>2
    #    print(seper)
    #    mask.transpose()
    continuous = data[:,mask]
    print(continuous.shape)
    z_sample,x_hat_1,x_hat_2,x_hat_3 = mnist_vae(continuous,gene_size,feed_dict)
    from sklearn.neighbors import KNeighborsClassifier 
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(continuous,range(len(continuous)))
    com = neigh.predict(x_hat_1)
    pre_1 = []
    i = 0
    for index,value in enumerate(mask):
    	if value == True:
    		pre_1.append(x_hat_1[:,i])
    		i = i+1
    	else:
    		tmp = data[com,index]
    		pre_1.append(tmp)
    com = neigh.predict(x_hat_2)
    pre_2 = []
    i = 0
    for index,value in enumerate(mask):
    	if value == True:
    		pre_2.append(x_hat_2[:,i])
    		i = i+1
    	else:
    		tmp = data[com,index]
    		pre_2.append(tmp)
    com = neigh.predict(x_hat_3)
    pre_3 = []
    i = 0
    for index,value in enumerate(mask):
    	if value == True:
    		pre_3.append(x_hat_3[:,i])
    		i = i+1
    	else:
    		tmp = data[com,index]
    		pre_3.append(tmp)
    print('we are using vae 6')
    print('this time we are generating')
    check = pd.value_counts(com)
    print(check.shape)
    pre_1 = np.array(pre_1)
    pre_2 = np.array(pre_2)
    pre_3 = np.array(pre_3)
#    print(pre.shape)
    return z_sample,pre_1.transpose(),pre_2.transpose(),pre_3.transpose()