# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 11:03:25 2017

@author: zhouying
"""

def random_walk(z,gene_size):
    import numpy as np
    sigma = np.var(z,axis=0)/np.power(z.shape[0],0.5)
#    random = np.random.normal(0,1,gene_size)
    z_sample = []
    k = 0
    j = 0
    while(k<gene_size):
        z_gene = z[j]-sigma[k]*np.random.normal(0,1)
        z_sample.append(z_gene)
        k = k+1
        j = k%z.shape[0]
        if k%gene_size == 0:
            np.random.shuffle(z)
    return z_sample
        