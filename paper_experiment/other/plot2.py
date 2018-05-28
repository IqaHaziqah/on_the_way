# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 09:20:24 2017

@author: zhouying
"""
import matplotlib.pyplot as plt
j = [i for i in range(data.shape[1])]
plt.figure(1)
plt.subplot(311)
for value in data:
    plt.plot(j,value)
plt.ylabel('real')

plt.subplot(312)
for value in ans:
    plt.plot(j,value)
plt.ylabel('predict')
plt.ylim((0,1))

plt.subplot(313)
cha = data-ans
for value in cha:
    plt.plot(j,value)
plt.ylabel('difference')
plt.savefig('distribution',dpi=300)