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
    z_sample,x_hat_1 = mnist_vae(continuous,gene_size,feed_dict)
    from sklearn.neighbors import KNeighborsClassifier 
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(continuous,range(len(continuous)))
    com = neigh.predict(x_hat_1)
    pre = []
    i = 0
    for index,value in enumerate(mask):
    	if value == True:
    		pre.append(x_hat_1[:,i])
    		i = i+1
    	else:
    		tmp = data[com,index]
    		pre.append(tmp)
    
    print('we are using vae 6')
    print('this time we are generating')
    check = pd.value_counts(com)
    print(check.shape)
    pre = np.array(pre)
#    print(pre.shape)
    return z_sample,pre.transpose()