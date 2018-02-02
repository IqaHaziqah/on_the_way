# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:05:30 2017

@author: zhouying
"""
def wanna_see(data,generate):
	import tensorflow as tf
	import os
	from tensorflow.contrib.tensorboard.plugins import projector
	

	logdir = os.path.join(os.path.abspath('.'),'event')


	
    

	embedding_var_data = tf.Variable(data,name='data')
	embedding_var_gene = tf.Variable(generate,name='generate')
	summary_writer = tf.summary.FileWriter(logdir)
	config = projector.ProjectorConfig()
	embedding = config.embeddings.add()
	embedding.tensor_name = embedding_var_data.name
	embedding2 = config.embeddings.add()
	embedding2.tensor_name = embedding_var_gene.name

#	path_for_metadata = os.path.join(logdir,'metadata.tsv')
#	embedding.metadata_path = path_for_metadata
#	embedding2.metadata_path = path_for_metadata
	projector.visualize_embeddings(summary_writer,config)
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.save(sess, os.path.join(logdir, "model.ckpt"), 1)