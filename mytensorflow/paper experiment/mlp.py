#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:17:57 2017

@author: zhouying
"""
#    输入为[归一化的feature，label],test_x
#    输出为预测的类标
def mlp(data,x_test):
    import tensorflow as tf
    from myutil import Dataset,onehot
    import numpy as np
    learning_rate = 0.001
    batch_size = 20
    epoch = 20
    input_dim = data.shape[1]-1
    hidden1 = 25
    hidden2 = 10
    output_dim = 2
    
#    x_train = data[:,0:input_dim]
    x_train = Dataset(data)
    
    
    x = tf.placeholder('float',shape = [None,input_dim])
    y = tf.placeholder('float',shape = [None,output_dim])
    keep_rate = tf.placeholder(tf.float32)
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial)
    
    hidden1_w = weight_variable([input_dim,hidden1])
    hidden1_b = bias_variable([hidden1])
    hidden1_o = tf.nn.relu(tf.matmul(x,hidden1_w)+hidden1_b)
    
    hidden2_w = weight_variable([hidden1,hidden2])
    hidden2_b = bias_variable([hidden2])
    hidden2_o = tf.nn.relu(tf.matmul(hidden1_o,hidden2_w)+hidden2_b)
    
    out_w = weight_variable([hidden2,output_dim])
    out_b = bias_variable([output_dim])
    y_pre = tf.nn.softmax(tf.matmul(hidden2_o,out_w)+out_b)
    
    loss = tf.reduce_mean(tf.pow(y-y_pre,2))
    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            total = int(data.shape[0]/batch_size)
            batch = x_train.next_batch(batch_size)
            label = batch[:,input_dim]
            lab = np.zeros(len(label),dtype = 'int')
            for k in range(len(label)):
                lab[k] = int(label[k])
            lab = onehot(lab,output_dim)
#            print(lab.shape)
            for j in range(total):
                _, cur_loss = sess.run([train_step,loss], feed_dict={x: batch[:,0:input_dim],y:lab,keep_rate:0.5})
                if j % 5 == 0:
                    print("Step {0} | Loss: {1}".format((i*total+j), cur_loss))
        classes = sess.run([y_pre],feed_dict ={x:x_test,keep_rate:1})
        classes = classes[0]
        classes = tf.arg_max(classes,1)
        la = classes.eval()
    
    return la 