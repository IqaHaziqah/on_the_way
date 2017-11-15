# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np



f = open("./CH1949BST.txt")  
line = f.readline().split() 
X = np.empty((0,6), dtype=int)
Y = np.empty((0,2), dtype=int)

#def nonlin(x,deriv=False):
#    if(deriv==True):
#        return x*(1-x)
#    return 1/(1+np.exp(-x))

#np.random.seed(1)
## randomly initialize our weights with mean 0
#syn0 = 2*np.random.random((6,4)) - 1
## print('syn0:\n', syn0)
#syn1 = 2*np.random.random((4,2)) - 1
# print('syn1:\n', syn1)
# while line:
if  line[0]=='66666':  
    if line[0]=='66666':
#        print line,
#        print '\n'
        recodersize=int(line[2])
        if recodersize>4:
            for i in range(recodersize):
                if i<4:
                    line = f.readline().split() #读一行
                    line = list(map(int, line))  #String转为Int
                    line[0]=(line[0]%1000000-line[0]%10000)/10000  #只保留年
#                     print line
                    Xtemp=np.row_stack((X,line))
                    X=Xtemp
                elif i>=4 and i<recodersize-4:
                    line = f.readline().split() #读一行
                    line = list(map(int, line))  #String转为Int
                    line[0]=(line[0]%1000000-line[0]%10000)/10000  #只保留年
                    Xtemp=np.row_stack((X,line))
                    X=Xtemp
                    result=[line[2],line[3]]
                    Ytemp=np.row_stack((Y,result))
                    Y=Ytemp
                else:
                    line = f.readline().split() #读一行
                    line = list(map(int, line))  #String转为Int
                    line[0]=(line[0]%1000000-line[0]%10000)/10000  #只保留年
                    result=[line[2],line[3]]
                    Ytemp=np.row_stack((Y,result))
                    Y=Ytemp
                            
#            print X
#            print('Y:\n', Y)
#            print '\n'
#            for j in xrange(10):
#                l0 = X
# 
##                 l1 = np.dot(l0,syn0)
##                 print('l1:\n',l1)
##                 l2 = np.dot(l1,syn1)
#                
#                l1 = nonlin(np.dot(l0,syn0))
#                print('l1:\n',l1)               
#                l2 = np.dot(l1,syn1)
#                print('l2:\n',l2)
#                
#
#                
#                # how much did we miss the target value?
#                l2_error = Y - l2
#                print('l2_error:\n',l2_error)
#                print "l2_error:" + str(np.mean(np.abs(l2_error)))
##                 if (j% 1) == 0:
##                     print "Error:" + str(np.mean(np.abs(l2_error)))
#
#                l2_delta = l2_error*l2
#                print "l2_delta:" + str(np.mean(np.abs(l2_delta)))
#                l1_error = l2_delta.dot(syn1.T)
#                print "l1_error:" + str(np.mean(np.abs(l1_error)))
#                l1_delta = l1_error * nonlin(l1,deriv=True) 
#                print "l1_delta:" + str(np.mean(np.abs(l1_delta)))               
##                 l2_delta = l2_error*nonlin(l2,deriv=True)
##                 l1_error = l2_delta.dot(syn1.T)
##                 l1_delta = l1_error * nonlin(l1,deriv=True)
#                syn1 += l1.T.dot(l2_delta)
#                syn0 += l0.T.dot(l1_delta)
        
    line = f.readline().split() 
      
f.close() 
print('X:\n', X)
print('Y:\n', Y)