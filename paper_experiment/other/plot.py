#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:25:01 2017

@author: zhouying
"""


import matplotlib.pyplot as plt
from pylab import *

for i in range(len(b_ne_1)):
    if y_test[i] ==1:
        plt.scatter(i,b_ne_1[i],c='b',marker='o')
        plt.scatter(i,b_po_1[i],c='r',marker='o')
        plt.scatter(i,ge_po_1[i],c='y',marker='o')
    else:
        plt.scatter(i,b_ne_1[i],c='b',marker='+')
        plt.scatter(i,b_po_1[i],c='r',marker='+')
        plt.scatter(i,ge_po_1[i],c='y',marker='+')
plt.show()

#plt.hist(b_ne_1)
#plt.hist(b_po_1)
#plt.hist(ge_po_1)
