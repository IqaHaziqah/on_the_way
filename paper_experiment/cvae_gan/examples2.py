# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:18:07 2018

@author: zhouying
"""

import tensorflow as tf  
import numpy as np
from numpy.random import RandomState  
  
if __name__ == "__main__":  
    #设置每次迭代时数据的大小  
    batch_size = 8  
    #定义输入和输出节点  
    #定义输入节点，设置输入的数据为1行2两列，设置None的目的是表示不确定输入数据的个数  
    #最后，我们会将整个的数据数据整合成一个矩阵  
    x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')  
    #定义输出节点，保证每个输入值所对应的输出值为1维的  
    y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')  
    #定义神经网络的前向传播过程  
    #定义参数，设置参数的大小为两行一列  
    w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))  
    #输入矩阵与参数进行加权和  
    y = tf.matmul(x,w1)  
    #定义预测多了和预测少了的损失的利润  
    loss_less = 5  
    loss_more = 1  
    #定义损失函数  
    loss = tf.reduce_sum(tf.where(tf.equal(y_,0),(y-y_)*loss_more,(y_-y)*loss_less))  
    #最小化损失函数  
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)  
    #随机生成一个数据集  
    rdm = RandomState(1)  
    #设置数据集的大小  
    dataset_size = 128  
    #产生输入数据  
    X = rdm.rand(dataset_size,2)  
    #定义输出y  
    ''''' 
    设置样本的输出值为，两个输入值相加并加上一个随机量。 
    还为输出值设置了一个-0.05~0.05的随机数噪音，达到模拟真实数据的效果。 
    '''  
#    Y = [[x1+x2 + rdm.rand()/5.0 - 0.05] for (x1,x2) in X]  
    Y = np.random.randint(0,2,[dataset_size,1])
    #训练神经网络  
    with tf.Session() as sess:  
        #初始化参数变量  
        init_op = tf.global_variables_initializer()  
        sess.run(init_op)  
        #设置迭代次数  
        STEPS = 5000  
        for i in range(STEPS):  
            start = (i * batch_size) % dataset_size  
            end = min(start + batch_size,dataset_size)  
            #训练模型  
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})  
        print(w1.eval())  