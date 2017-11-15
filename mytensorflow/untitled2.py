#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 11:25:13 2017

@author: zhouying
"""

import tensorflow as tf
import numpy as np
#import math
#import os 
#mnist.train.next_batch 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot= True)
help(mnist.train.next_batch)
input_units = 784
hidden1 = 100
hidden2 = 50
output_units = 10

x = tf.placeholder('float',[None,input_units])
y = tf.placeholder('float',[None,output_units])

#hidden1
with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([input_units,hidden1]),
                name='weights')
    biaes = tf.Variable(tf.zeros([hidden1]),name='bias')
   
    hidden11 = tf.nn.relu(tf.matmul(x,weights)+biaes)
    
#hidden2
with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([hidden1,hidden2]),name = 'weights')
    biaes = tf.Variable(tf.zeros([hidden2]),name = 'bias')
    hidden22 = tf.nn.relu(tf.matmul(hidden11,weights)+biaes)
    
#output
with tf.name_scope('out'):
    weights = tf.Variable(tf.truncated_normal([hidden2,output_units]),name = 'weights')
    biaes = tf.Variable(tf.zeros([output_units]),name = 'bias')
    output = tf.matmul(hidden22,weights)+biaes
    
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits=output))
learning_rate = 0.001
batchsize = 100

optermizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optermizer.minimize(loss)
init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(mnist.test.labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


with tf.Session() as sess:
    sess.run(init_op)
    
    total_batch = int(mnist.train.num_examples/batchsize)
    for i in range(total_batch):
        batch_x,batch_y = mnist.train.next_batch(batchsize)
#        print(batch_x.shape)
        _,loss_value = sess.run([train_op,loss],feed_dict={x:batch_x,y:batch_y})
    print(loss_value)
    pre = sess.run([accuracy],feed_dict = {x:mnist.test.images})
    print(pre)
#acc = tf.equal(tf.maximum(pre,1),mnist.test.labels)/100
#correct_prediction = tf.equal(tf.argmax(np.array(pre), 1), tf.argmax(x, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))