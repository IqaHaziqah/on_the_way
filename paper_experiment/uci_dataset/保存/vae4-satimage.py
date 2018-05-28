#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:38:51 2017

@author: zhouying
"""
"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
import os

logdir = os.path.join(os.path.abspath('.'),'event4')
def variable_summaries(var,var_name):
    import tensorflow as tf
    with tf.name_scope(var_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#          print(var_name,tf.square(stddev))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def wanna_see(data,generate,label=[]):
    import tensorflow as tf
    import os
    from tensorflow.contrib.tensorboard.plugins import projector
    import numpy as np
    
    see = np.concatenate([data,generate],axis=0)
    embedding_var = tf.Variable(see,name='wanna_see')
    summary_writer = tf.summary.FileWriter(logdir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    if label==[]:
        labels=np.concatenate([np.zeros(data.shape[0]),np.ones(generate.shape[0])],axis=0)
    else:
        next = np.argmax(label)+1
        labels = np.concatenate([label,np.ones(generate.shape[0])*next],axis=0)
    path_for_metadata = os.path.join(logdir,'metadata.tsv')
    embedding.metadata_path = path_for_metadata
    projector.visualize_embeddings(summary_writer,config)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(logdir, "model.ckpt"), 1)
    with open(path_for_metadata,'w') as f:
        f.write('Index\tLabel\n')
        for index,value in enumerate(labels):
            f.write('%d\t%d\n'%(index,value))
def weight_variable(shape):
    import tensorflow as tf
    import numpy as np
#    initial = tf.truncated_normal(shape, stddev=0.01)*0.001
#    D,H = shape
#    import numpy as np
#    initial = 0.0001*np.random.randn(D,H)
#    /np.sqrt(D)
    n,out = shape
    initial = np.zeros(shape,dtype='float32')
    for j in range(out):
        initial[:,j] = np.random.randn(n)/np.sqrt(n)
    return tf.Variable(initial)

def bias_variable(shape):
    import tensorflow as tf
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def mnist_vae(data,gene_size,feed_dict):
    import tensorflow as tf
    from myutil2 import Dataset
    import numpy as np
    from scipy.stats import norm
    from myutil2 import random_walk,xavier_init
    from keras import metrics
    mnist = Dataset(data)
    input_dim = data.shape[1]
    hidden_encoder_dim = feed_dict['hidden_encoder_dim']
    hidden_decoder_dim = feed_dict['hidden_decoder_dim']
    latent_dim = feed_dict['latent_dim']
    lam = feed_dict['lam']
    epochs = feed_dict['epochs']
    batch_size = feed_dict['batch_size']
    learning_rate = feed_dict['learning_rate']
    ran_walk = feed_dict['ran_walk']
#    trade_off = feed_dict['trade_off']
    check = feed_dict['check']
    
    tf.set_random_seed(1)
    
    

    with tf.name_scope('inputs'):
        x = tf.placeholder("float", shape=[None, input_dim],name='input_x')
        input_z = tf.placeholder("float",shape = [None,latent_dim],name='input_z')
        epsilon = tf.placeholder('float',shape = [None,latent_dim],name='eplison')
        
    l2_loss = tf.constant(0.0)
    
    with tf.name_scope('encode_hidden'):
        W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim])
    #    W_encoder_input_hidden = xavier_init(input_dim,hidden_encoder_dim)
        b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
        put1 = tf.matmul(x, W_encoder_input_hidden)
        if check == True:
            variable_summaries(W_encoder_input_hidden, 'W_encoder_input_hidden')
#            variable_summaries(b_encoder_input_hidden, 'b_encoder_input_hidden')
            
            variable_summaries(put1,'put1')
#    with tf.name_scope('variance'):
        
    
    # Hidden layer encoder
        hidden_encoder = tf.nn.tanh(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)
    l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)
    with tf.name_scope('encode_mu'):
        W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
    #    W_encoder_hidden_mu = xavier_init(hidden_encoder_dim,latent_dim)
        b_encoder_hidden_mu = bias_variable([latent_dim])
        mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu
        if check == True:
            variable_summaries(W_encoder_hidden_mu, 'W_encoder_hidden_mu')
            variable_summaries(mu_encoder-b_encoder_hidden_mu,'input21')
#            variable_summaries(b_encoder_hidden_mu, 'b_encoder_hidden_mu')
#    with tf.name_scope('variance'):
        
          
        # Mu encoder
        
        
    l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)
    with tf.name_scope('encode_logvar'):
        W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
    #    W_encoder_hidden_logvar = xavier_init(hidden_encoder_dim,latent_dim)
        b_encoder_hidden_logvar = bias_variable([latent_dim])
        logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar
        if check == True:
            variable_summaries(W_encoder_hidden_logvar, 'W_encoder_hidden_logvar')
            variable_summaries(logvar_encoder-b_encoder_hidden_logvar, 'input22')
#            variable_summaries(b_encoder_hidden_logvar, 'b_encoder_hidden_logvar')
#    with tf.name_scope('variance'):
        
    # Sigma encoder
        
    l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)
# Sample epsilon
    

# Sample latent variable
    std_encoder = tf.exp(0.5 * logvar_encoder)
    z = mu_encoder + tf.multiply(std_encoder, epsilon)

    with tf.name_scope('deocde_z'):
        W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
    #    W_decoder_z_hidden = xavier_init(latent_dim,hidden_decoder_dim)
        b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
        input3 = tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden
        hidden_decoder = tf.nn.tanh(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)
        if check == True:
            variable_summaries(W_decoder_z_hidden, 'W_decoder_z_hidden')
            variable_summaries(tf.matmul(z, W_decoder_z_hidden), 'input3')
#            variable_summaries(b_decoder_z_hidden, 'b_decoder_z_hidden')
#    with tf.name_scope('variance'):
        
    # Hidden layer decoder
        
    l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)
    
#    with tf.name_scope('decode_logvar'):
#        dl_weights = weight_variable([hidden_decoder,input_dim])
#        dl_bias = bias_variable([input_dim])
#        decoder_logvar = tf.matmul(hidden_decoder,dl_weights)+dl_bias
#        if check == True:
#            variable_summaries(decoder_logvar, 'decoder_logvar')
#    
#    with tf.name_scope('decode_mu'):
#        dm_weights = weight_variable([hidden_decoder,input_dim])
#        dm_bias = bias_variable([input_dim])
#        decoder_mu = tf.matmul(hidden_decoder,dm_weights)+dm_bias
#        if check == True:
#            variable_summaries(decoder_mu, 'decoder_mu')
    
    with tf.name_scope('decode_hidden'):
        
        W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
    #    W_decoder_hidden_reconstruction = xavier_init(hidden_decoder_dim, input_dim)
        b_decoder_hidden_reconstruction = bias_variable([input_dim])   
        x_hat = (tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction)
        
        if check == True:
            variable_summaries(W_decoder_hidden_reconstruction, 'W_decoder_hidden_reconstruction')
#            tf.summary.scalar('input4',x_hat)
#            variable_summaries(b_decoder_hidden_reconstruction, 'b_decoder_hidden_reconstruction')
    with tf.name_scope('variance'):
        tf.summary.scalar('input4',tf.reduce_mean(tf.square(x_hat - tf.reduce_mean(x_hat))))
        tf.summary.scalar('input1',tf.reduce_mean(tf.square(put1 - tf.reduce_mean(put1))))
        tf.summary.scalar('input3',tf.reduce_mean(tf.square(input3 - tf.reduce_mean(input3))))
        tf.summary.scalar('input22',tf.reduce_mean(tf.square(logvar_encoder - tf.reduce_mean(logvar_encoder))))
        tf.summary.scalar('input2',tf.reduce_mean(tf.square(mu_encoder - tf.reduce_mean(mu_encoder))))
        
    l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)
    KLD = -0.5*tf.reduce_sum(1+logvar_encoder-tf.square(mu_encoder)-tf.exp(logvar_encoder),axis=-1)
    kld = tf.reduce_mean(KLD)
    
#    BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x), reduction_indices=1)
#    BCE = tf.reduce_sum(tf.abs(x_hat-x))
    BCE = tf.reduce_sum(tf.pow(x_hat-x,2),reduction_indices=1)
#    BCE = input_dim * metrics.binary_crossentropy(x, x_hat)
    loss = tf.reduce_mean(BCE + KLD)
    regularized_loss = loss + lam * l2_loss
    if check == True:
        tf.summary.scalar('unregularied_loss',loss)
        tf.summary.scalar('lowerbound',kld)
        tf.summary.scalar('binary_crossentropy',tf.reduce_mean(BCE))
    
    
    grad = tf.gradients(regularized_loss,[W_decoder_z_hidden,W_decoder_hidden_reconstruction])
#    loss_summ = tf.summary.scalar("lowerbound", loss)
    train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=0.9).minimize(regularized_loss)
#    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(regularized_loss)
    merged = tf.summary.merge_all()
    hidden_decoder_1 = tf.nn.tanh(tf.matmul(input_z, W_decoder_z_hidden) + b_decoder_z_hidden)
    x_hat_1 = (tf.matmul(hidden_decoder_1, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction)
    
    
#    if tf.gfile.Exists(logdir):
#        tf.gfile.DeleteRecursively(logdir)
        

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(logdir,sess.graph)
        total = int(data.shape[0]/batch_size)+1
#        projector.visualize_embeddings(writer,config)
        for i in range(epochs):
            for j in range(total):
#                _,train = sess.run([train_step,merged], feed_dict={x: mnist.next_batch(batch_size)[0]})
                batch = mnist.next_batch(batch_size)[0]
                e = np.random.normal(size=[1,latent_dim])
                sess.run(train_step,feed_dict={x:batch,epsilon:e})
            if check == True:
                result = sess.run(merged,feed_dict={x:data,epsilon:e})
                writer.add_summary(result,i)

            if i*total == 25000:
                learning_rate = 1e-4
#        saver.save(sess,os.path.join(logdir,'model.ckpt'),1)
        if feed_dict['check'] == True:
            z_sample = sess.run(z,feed_dict={x:data,epsilon:e})
            
        elif ran_walk == True:
            zz = sess.run([z],feed_dict={x:data,epsilon:e})#
#        print(zz.shape)
            z_sample = random_walk(zz,gene_size)
        else:
            z_sample = np.random.normal(size=[gene_size,latent_dim])
        
        x_hat_1 = sess.run(x_hat_1,feed_dict = {input_z:z_sample,epsilon:e})
        k,l,mw,lw=sess.run([kld,loss,W_encoder_hidden_mu,W_encoder_hidden_logvar],feed_dict={x:data,epsilon:e})
        print('lower_bound,mean_loss',k,l)
#        print(mw)
#        print(lw)
        print(sess.run(grad,feed_dict={x:data,epsilon:e}))
        print(sess.run(W_decoder_hidden_reconstruction))
#        wanna_see(data,x_hat_1)
        writer.close()
        sess.close()
    tf.reset_default_graph()
    wanna_see(data,x_hat_1)
    return z_sample,x_hat_1