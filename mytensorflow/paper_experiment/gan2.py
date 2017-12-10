#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:51:18 2017

@author: zhouying
"""
def gan(data,gene_size):
	import numpy as np
	import tensorflow as tf
	from myutil import Dataset
	
	input_size = data.shape[1]
	hidden1 = 25
	hidden2 = 20
	noise_size = 10
	batch_size = 20
	mnist = Dataset(data)
	
	def dense(inputs,shape,name,bn=False,act_fun=None):
		W = tf.get_variable(name+".w",initializer=tf.random_normal(shape=shape))
		tf.summary.histogram(name+'.w',W)
		b = tf.get_variable(name+".b",initializer=tf.zeros((1,shape[-1]))+0.1) 
		y = tf.add(tf.matmul(inputs,W),b)
		tf.summary.histogram(name+'.b',b)
		def batch_normalization(inputs,out_size,name,axes=0):
			mean,var = tf.nn.moments(inputs,axes=[axes])
			scale = tf.get_variable(name=name+".scale",initializer=tf.ones([out_size]))
			offset = tf.get_variable(name = name+".shift",initializer=tf.zeros([out_size]))
			epsilon = 0.001
			return tf.nn.batch_normalization(inputs,mean,var,offset,scale,epsilon,name=name+".bn")
		if bn:
			y = batch_normalization(y,shape[1],name=name+".bn")
		if act_fun:
			y = act_fun(y)
		tf.summary.histogram(name+'.y',y)
		return y
	
	def D(inputs,name,reuse=False):
		with tf.variable_scope(name,reuse=reuse):
			l1 = dense(inputs,[input_size,hidden1],name='relu1',act_fun=tf.nn.relu)
#			l2 = dense(l1,[hidden1,hidden2],name='relu2',act_fun=tf.nn.relu)
			y = dense(l1,[hidden1,1],name='output')
			return y
	
	def G(inputs,name,reuse=False):
		with tf.variable_scope(name,reuse=reuse):
			l1 = dense(inputs,[noise_size,hidden1],name='relu1',act_fun=tf.nn.relu)
#			l2 = dense(l1,[hidden2,hidden1],name='relu2',act_fun=tf.nn.relu)
			y  = dense(l1,[hidden1,input_size],name='output',act_fun=tf.nn.sigmoid)
			return y
			
	z = tf.placeholder(tf.float32,[None,noise_size],name='noise')
	x = tf.placeholder(tf.float32,[None,input_size],name='image')
	
	real_out = D(x,'D')
	gen = G(z,'G')
	fake_out = D(gen,'D',reuse=True)
	
	vars = tf.trainable_variables()
	
	d_para = [var for var in vars if var.name.startswith('D')]
	g_para = [var for var in vars if var.name.startswith('G')]
	
	d_clip = [tf.assign(var,tf.clip_by_value(var,-0.01,0.01)) for var in d_para]
	d_clip = tf.group(*d_clip)
	
	wd = tf.reduce_mean(real_out)-tf.reduce_mean(fake_out)
	d_loss = tf.reduce_mean(fake_out)-tf.reduce_mean(real_out)
	g_loss = tf.reduce_mean(fake_out)
	tf.summary.scalar('wd',wd)
	tf.summary.scalar('d_loss',d_loss)
	tf.summary.scalar('g_loss',g_loss)
	d_opt = tf.train.RMSPropOptimizer(1e-3).minimize(
		d_loss,
		global_step=tf.Variable(0),
		var_list = d_para)
	g_opt = tf.train.RMSPropOptimizer(1e-3).minimize(
		g_loss,
		global_step = tf.Variable(0),
		var_list = g_para)
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('/home/zhouying/log',sess.graph)
	
	CRITICAL_NUM = 1
	for step in range(10*300):
		if step <25 or step%500==0:
			critical_num = 100
		else:
			critical_num = CRITICAL_NUM
		for ep in range(critical_num):
			noise = np.random.normal(size=(batch_size,noise_size))
			batch_xs = mnist.next_batch(batch_size)
			_,d_loss_v,_ = sess.run([d_opt,d_loss,d_clip],feed_dict={
				x:batch_xs,
				z:noise
				})
		for ep in range(1):
			noise = np.random.normal(size=(batch_size,noise_size))
			_,g_loss_v = sess.run([g_opt,g_loss],feed_dict={
				z:noise
				})
#		print("step:%d D-loss:%.4f G-loss:%.4f"%(step+1,d_loss_v,g_loss_v))
#		if step%50 == 0:
#			result = sess.run(merged,feed_dict={x:batch_xs,z:noise})
#			writer.add_summary(result,step)
		if step%100 ==99:
			batch_xs = mnist.next_batch(batch_size)
			noise = np.random.normal(size=(batch_size,noise_size))
			mlp_v = sess.run(wd,feed_dict={
				x:batch_xs,
				z:noise
				})
			print("####step%d Wd:%.4f####"%(step+1,mlp_v))
	noise = np.random.normal(size=(gene_size,noise_size))	
	gene = sess.run(gen,feed_dict={z:noise})
    
	writer.close
	sess.close
	return gene