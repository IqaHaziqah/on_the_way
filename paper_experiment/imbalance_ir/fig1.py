# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:05:21 2018

@author: zhouying
"""

import numpy as np
import matplotlib.pyplot as plt
#f11 = np.random.uniform(0,0.01,10)
#f12 = np.random.uniform(0,0.02,10)
#f21 = np.random.uniform(0,0.01,10)
#f22 = np.random.uniform(0,0.02,10)
plt.scatter(f11,f21,c='#000000',marker='o',label='class1')
plt.scatter(f12,f22,c='r',marker='*',label='class2')
plt.legend(loc='upper left')
plt.xlabel('f1')
plt.ylabel('f2')
plt.xticks([])
plt.yticks([])
plt.savefig('figb',dpi=300)