# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:16:03 2018

@author: zhouying
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as scio

i = np.array([j for j in range(100)])
sita = i*math.pi/16
r = 6.5*(104-i)/104
x = r*np.sin(sita)
y = r*np.cos(sita)

ax1 = plt.subplot(111)
ax1.scatter(x,y,c='r')
ax1.scatter(-x,-y,c='b')

plt.savefig('螺旋结构.png',dpi=300)
data = np.concatenate((np.concatenate(([x],[-x]),axis=1),np.concatenate(([y],[-y]),axis=1)),axis=0)
label = np.concatenate((np.ons(100),np.zeros(100)),axis=0)
scio.savemat('..\MNIST_data\luoxuan.mat',{})
ans = np.concatenate(([x],[y]),axis=0)
