#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 20:29:20 2017

@author: zhouying
"""

def trysdae(data,hidden1,hidden2):
    import tensorflow as tf
    import numpy as np
    from myutil import randomselect,xavier_init
    
    input_units = data.shape[1]
    x = tf.placeholder('float',[None,input_units])
    #hidden1
    with tf.name_scope('hidden_layer1'):
        encoder_weights = tf.Variable(xavier_init([input_units,hidden1],name='encoder_weights'))
        encoder_bias = tf.Variable(tf.zeros(hidden1),name='encoder_bias')
        decoder_weights = tf.Variable(xavier_init([hidden1,input_units],name='decoder_weights'))
        decoder_bias = tf.Variable(tf.zeros(input_units),name='decoder_bias')
        z1 = tf.nn.sigmoid(tf.matmul(x,encoder_weights)+encoder_bias)
        output1 = tf.nn.sigmoid(tf.matmul(z1,decoder_weights)+decoder_bias)
        loss1 = tf.reduce_mean(tf.pow(x-output1,2))
        
    with tf.name_scope('hidden_layer2'):
        encoder_weights = tf.Variable(xavier_init([hidden1,hidden2],name='encoder_weights'))
        encoder_bias = tf.Variable(tf.zeros(hidden2),name='encoder_bias')
        decoder_weights = tf.Variable(xavier_init([hidden2,hidden1],name='decoder_weights'))
        decoder_bias = tf.Variable(tf.zeros(hidden1),name='decoder_bias')
        z2 = tf.nn.sigmoid(tf.matmul(x,encoder_weights)+encoder_bias)
        output2 = tf.nn.sigmoid(tf.matmul(z2,decoder_weights)+decoder_bias)
        loss2 = tf.reduce_mean(tf.pow(x-output2,2))
        
    with tf.name_scope('reuse'):
         
    init_op = tf.global_variables_initializer()
    optermizer = tf.train.AdamOptimizer()
    train_op = optermizer.minimize(loss1)
    train_op2 = optermizer.minimize(loss2)
    train_op3 = optermizer.minimize(loss3)
    
    
