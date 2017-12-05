# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:30:30 2017

@author: zhouying
"""

import numpy as np
import tensorflow as tf

xx = np.random.standard_normal(10)
yy = 2*xx+0.1

with tf.name_scope('input'):
    x = tf.placeholder('float',shape=[None],name='x')
    tf.summary.scalar('input_x',x)
    y = tf.placeholder('float',shape=[None],name='y')
    tf.summary.scalar('input_y',y)
    
with tf.name_scope('para'):
    W = tf.Variable([1],dtype='float32')
    tf.summary.scalar('weights',W)
    b = tf.zeros([1])
    tf.summary.scalar('bias',b)
    
output = tf.nn.relu(tf.multiply(x,W)+b)
loss = tf.reduce_mean(tf.pow((y-output),2),reduction_indices=1)

optimizer = tf.train.AdamOptimizer(loss)
merged=tf.summary.merge_all()

sess = tf.Session()
writer = tf.summary.FileWriter('.\\events\\',sess.graph)
sess.run(tf.global_variables_initializer())
for i in range(10):
    print(sess.run(output,feed_dict={x:xx,y:yy}))
#    su,_=sess.run([merged,optimizer],feed_dict={x:xx,y:yy})
    sess.run(optimizer,feed_dict={x:xx,y:yy})
    
#    writer.add_summary(su,i)
    
    
