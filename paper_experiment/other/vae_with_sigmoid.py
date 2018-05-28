# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 20:34:10 2018

@author: zhouying
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from vae4 import weight_variable,bias_variable,variable_summaries,wanna_see
from keras import metrics
mnist = input_data.read_data_sets('..\MNIST_data')
hidden_dim = 500
input_dim = 784
latent_dim = 5
learning_rate = 1e-3
total_step = int(1e4)
batch_size = 100

with tf.name_scope('input'):
    x = tf.placeholder('float32',[None,input_dim],name='x')
    input_z = tf.placeholder('float32',[None,latent_dim],name='input_z')

with tf.name_scope('encoder'):
    e_weight = weight_variable([input_dim,hidden_dim])
    e_bias = bias_variable([hidden_dim])
    e_output = tf.nn.relu(tf.matmul(x,e_weight)+e_bias)
    variable_summaries(e_weight,'e_weight')
    variable_summaries(e_bias,'e_bias')
    
with tf.name_scope('mu'):
    m_weight = weight_variable([hidden_dim,latent_dim])
    m_bias = bias_variable([latent_dim])
    m_output = (tf.matmul(e_output,m_weight)+m_bias)
    variable_summaries(m_weight,'m_weight')
    variable_summaries(m_bias,'m_bias')
    
with tf.name_scope('logvar'):
    l_weight = weight_variable([hidden_dim,latent_dim])
    l_bias = bias_variable([latent_dim])
    l_output = (tf.matmul(e_output,l_weight)+l_bias)
    variable_summaries(l_weight,'l_weight')
    variable_summaries(l_bias,'l_bias')

epsilon = tf.random_normal([latent_dim])
with tf.name_scope('z'):
    z = m_output+tf.multiply(epsilon,tf.exp(l_output/2))
    

with tf.name_scope('de_z'):
    de_weights = weight_variable([latent_dim,hidden_dim])
    de_bias = bias_variable([hidden_dim])
    variable_summaries(de_weights,'de_weights')
    variable_summaries(de_bias,'de_bias')
    de_output = tf.nn.relu(tf.matmul(z,de_weights)+de_bias)
    
    
with tf.name_scope('decoder'):
    d_weights = weight_variable([hidden_dim,input_dim])
    d_bias = bias_variable([input_dim])
    variable_summaries(d_weights,'d_weights')
    variable_summaries(d_bias,'d_bias')
    d_output = tf.nn.sigmoid(tf.matmul(de_output,d_weights)+d_bias)
    variable_summaries(d_output,'output')
    
    
with tf.name_scope('loss'):
    kld = -0.5*tf.reduce_sum(1+l_output-tf.square(m_output)-tf.exp(l_output),axis=-1)
    bce = input_dim*metrics.binary_crossentropy(x,d_output)
    total = tf.reduce_mean(kld+bce)
    tf.summary.scalar('bce',tf.reduce_mean(bce))
    tf.summary.scalar('kld',tf.reduce_mean(kld))
    tf.summary.scalar('total_loss',total)
    
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(total)
merged = tf.summary.merge_all()


logdir = '.\\event10\\'
#if tf.gfile.Exists(logdir):
#    tf.gfile.DeleteRecursively(logdir)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logdir,sess.graph)
    for i in range(total_step):
        
        batch = mnist.train.next_batch(batch_size)[0]
        batch = batch/255
        
        sess.run(train,feed_dict={x:batch})
        
        if i%100 ==0:
            result = sess.run(merged,feed_dict={x:batch})
            writer.add_summary(result,i)
writer.close()
sess.close()
tf.reset_default_graph()        
