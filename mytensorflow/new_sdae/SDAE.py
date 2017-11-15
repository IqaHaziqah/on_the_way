#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:56:48 2017

@author: zhouying
"""
def mysdae(data,stack_size = 2,hidden_size = [25,10]):
#from __future__ import division, print_function, absolute_import
    import numpy as np
    import tensorflow as tf
    import AdditiveGaussianNoiseAutoencoder as an
    

#定义训练参数
    training_epochs = 10
    batch_size = 10
    n_input = data.shape[1]
#    stack_size = stack_size
#    display_step = 1
#    stack_size = 3  #栈中包含3个ae
#    hidden_size = [20, 20, 20]
#    input_n_size = [3, 200, 200]

    def get_random_block_from_data(data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]

#建立sdae图
    sdae = []
    for i in range(stack_size):
        if i == 0:
            ae = an.AdditiveGaussianNoiseAutoencoder(n_input = n_input,n_hidden = hidden_size[i])
                                                  
#                                                   )
            ae._initialize_weights()
            sdae.append(ae)
        else:
            ae = an.AdditiveGaussianNoiseAutoencoder(n_input=hidden_size[i-1],
                                                  n_hidden=hidden_size[i],
                                                  transfer_function=tf.nn.softplus,
                                                  optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                  scale=0.01)
            ae._initialize_weights()
            sdae.append(ae)

		
#    W = []
#    b = []
    Hidden_feature = [] #保存每个ae的特征
    X_train = np.array([0])
    for j in range(stack_size):
    #输入
        if j == 0:
            X_train = data
#        X_test = np.array(pd.test_set)
        else:
            X_train_pre = X_train
            print(j,)
            print(X_train_pre.shape)
            X_train = sdae[j-1].transform(X_train_pre)
            print (X_train.shape)
            Hidden_feature.append(X_train)
	
	#训练
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X_train.shape[0] / batch_size)
        # Loop over all batches
            for k in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)

            # Fit training using batch data
                cost = sdae[j].partial_fit(batch_xs)
            # Compute average loss
                avg_cost += cost / X_train.shape[0] * batch_size

        # Display logs per epoch step
        #if epoch % display_step == 0:
#            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
	
	#保存每个ae的参数
#        weight = sdae[j].getWeights()
    #print (weight)
#        W.append(weight)
#        W.append(sdae[j].getallWeights())
#        b.append(sdae[j].getBiases())
#    return W
#
    for j in range(stack_size):
        if j == 0:
            x_input = data
        x_out = sdae[j].transform(x_input)
        print('x_out',x_out.shape)
        x_input = x_out
#        x_out = sdae[j].retransform(x_input)
    for j in range(stack_size):
        i = stack_size-j-1
        x_input = sdae[i].generate(x_input)
        print('x_reconstruction shape',x_input.shape)
#        x_input = x_out[i]
    return (x_input-data)
