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
def weight_variable(shape,ini):
    import tensorflow as tf
    n,out = shape
    if ini ==1:
        initial = tf.truncated_normal(shape, stddev=0.001)
    elif ini ==2:
        initial = tf.truncated_normal(shape,stddev=2.0/n)
    elif ini == 3:
        import numpy as np
        b = 1.0*np.sqrt(6.0/(n+out))
        initial = np.random.uniform(low=-b,high=b,size=shape)
    else:
        import numpy as np
        initial = np.zeros(shape,dtype='float32')
        for j in range(out):
            initial[:,j]=np.random(n)/np.sqrt(n)
    return tf.Variable(initial,dtype='float32')

def bias_variable(shape):
    import tensorflow as tf
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#def myoptimize(learning_rate):
#    import tensorflow as tf
    



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
    ini = feed_dict['initial']
    ACT = feed_dict['activation']
    OPT = feed_dict['optimizer']
    normal = feed_dict['norm']
    decay = feed_dict['decay']
    
    tf.set_random_seed(1)
    
    

    with tf.name_scope('inputs'):
        x = tf.placeholder("float", shape=[None, input_dim],name='input_x')
        input_z = tf.placeholder("float",shape = [None,latent_dim],name='input_z')
        
    l2_loss = tf.constant(0.0)
    
    with tf.name_scope('encode_hidden'):
        W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim],ini)
    #    W_encoder_input_hidden = xavier_init(input_dim,hidden_encoder_dim)
        b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
        input1 = tf.matmul(x,W_encoder_input_hidden)+b_encoder_input_hidden
        if normal:
            fc_mean,fc_var = tf.nn.moments(input1,axes=0)
            scale = tf.Variable(tf.ones([hidden_encoder_dim]))
            shift = tf.Variable(tf.zeros([hidden_encoder_dim]))
            eps = 0.001
            ema = tf.train.ExponentialMovingAverage(decay=decay)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean,fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean),tf.identity(fc_var)
            mean,var = mean_var_with_update()
            input1 = tf.nn.batch_normalization(input1,mean,var,shift,scale,eps)
        if check == True:
            variable_summaries(W_encoder_input_hidden, 'W_encoder_input_hidden')
#            variable_summaries(b_encoder_input_hidden, 'b_encoder_input_hidden')
    
    
    # Hidden layer encoder
        hidden_encoder = ACT(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)
    l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)
    with tf.name_scope('encode_mu'):
        W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim],ini)
    #    W_encoder_hidden_mu = xavier_init(hidden_encoder_dim,latent_dim)
        b_encoder_hidden_mu = bias_variable([latent_dim])
        
        if check == True:
            variable_summaries(W_encoder_hidden_mu, 'W_encoder_hidden_mu')
#            variable_summaries(b_encoder_hidden_mu, 'b_encoder_hidden_mu')
            
        # Mu encoder
        mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu
    l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)
    with tf.name_scope('encode_logvar'):
        W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim],ini)
    #    W_encoder_hidden_logvar = xavier_init(hidden_encoder_dim,latent_dim)
        b_encoder_hidden_logvar = bias_variable([latent_dim])
        
        if check == True:
            variable_summaries(W_encoder_hidden_logvar, 'W_encoder_hidden_logvar')
#            variable_summaries(b_encoder_hidden_logvar, 'b_encoder_hidden_logvar')
    # Sigma encoder
        logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar
    l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)
# Sample epsilon
    epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable
    std_encoder = tf.exp(0.5 * logvar_encoder)
    z = mu_encoder + tf.multiply(std_encoder, epsilon)

    with tf.name_scope('deocde_z'):
        W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim],ini)
    #    W_decoder_z_hidden = xavier_init(latent_dim,hidden_decoder_dim)
        b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
        
        if check == True:
            variable_summaries(W_decoder_z_hidden, 'W_decoder_z_hidden')
#            variable_summaries(b_decoder_z_hidden, 'b_decoder_z_hidden')
    # Hidden layer decoder
        hidden_decoder = ACT(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)
    l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)
    with tf.name_scope('decode_hidden'):
        W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim],ini)
    #    W_decoder_hidden_reconstruction = xavier_init(hidden_decoder_dim, input_dim)
        b_decoder_hidden_reconstruction = bias_variable([input_dim])    
        if check == True:
            variable_summaries(W_decoder_hidden_reconstruction, 'W_decoder_hidden_reconstruction')
#            variable_summaries(b_decoder_hidden_reconstruction, 'b_decoder_hidden_reconstruction')
#    KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)
#    KLD = 0
    l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)
#    KLD = 0.5*tf.reduce_sum(tf.square(mu_encoder)+tf.square(logvar_encoder)-tf.log(tf.square(logvar_encoder))-1,1)
    KLD = -0.5*tf.reduce_sum(1+logvar_encoder-tf.square(mu_encoder)-tf.exp(logvar_encoder),axis=-1)
    kld = tf.reduce_mean(KLD)
    x_hat = (tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction)
#    BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x), reduction_indices=1)
#    BCE = tf.reduce_sum(tf.abs(x_hat-x))
    BCE = tf.reduce_sum(tf.pow(x_hat-x,2),reduction_indices=1)
#    BCE = input_dim * metrics.binary_crossentropy(x, x_hat)
    loss = tf.reduce_mean(BCE + KLD)
    regularized_loss = loss + lam * l2_loss
#    loss = tf.reduce_mean(BCE+KLD)
    if check == True:
        tf.summary.scalar('unregularied_loss',loss)
        tf.summary.scalar('lowerbound',kld)
        tf.summary.scalar('binary_crossentropy',tf.reduce_mean(BCE))
    
    
    grad = tf.gradients(regularized_loss,[W_decoder_hidden_reconstruction,W_decoder_z_hidden])
#    loss_summ = tf.summary.scalar("lowerbound", loss)
    train_step = OPT(learning_rate).minimize(regularized_loss)
    if OPT == tf.train.RMSPropOptimizer:
        train_step = OPT(learning_rate=learning_rate,decay=decay).minimize(regularized_loss)
    
    merged = tf.summary.merge_all()
    hidden_decoder_1 = ACT(tf.matmul(input_z, W_decoder_z_hidden) + b_decoder_z_hidden)
    if OPT == tf.train.RMSPropOptimizer:
        train_step = OPT(learning_rate,decay=decay).minimize(regularized_loss)
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
            
                sess.run(train_step,feed_dict={x:batch})
            
#            result = sess.run(merged,feed_dict={x:data})
#            writer.add_summary(result,i)
#                pass
#                cur_loss = sess.run(loss,feed_dict={x:batch})
#                print("Step {0} | Loss: {1}".format(i,cur_loss))
#                writer.add_summary(train)
#                if j % 5 == 0:
#                    print("Step {0} | Loss: {1}".format((i*total+j), cur_loss))
#                    print("Step {0} | kld: {1}".format((i*total+j), cur_kld))
            if i*total == 25000:
                learning_rate = 1e-4
#        saver.save(sess,os.path.join(logdir,'model.ckpt'),1)
        if feed_dict['check'] == True:
            z_sample = sess.run(z,feed_dict={x:data})
            
        elif ran_walk == True:
            zz = sess.run([z],feed_dict={x:data})#
#        print(zz.shape)
            z_sample = random_walk(zz,gene_size)
        else:
#            miu,sigma=sess.run([mu_encoder,std_encoder],feed_dict={x:data})
#            miu = np.mean(miu,axis=0)
#            sigma = np.mean(sigma,axis=0)
#            ep = np.random.randn(gene_size,latent_dim)
#            z_sample = miu+sigma*ep
            z_sample = np.random.normal(size=[gene_size,latent_dim])
       
        x_hat_1 = sess.run(x_hat_1,feed_dict = {input_z:z_sample})
        k,l,mw,lw=sess.run([kld,loss,W_encoder_hidden_mu,W_encoder_hidden_logvar],feed_dict={x:data})
        print('lower_bound,mean_loss',k,l)
#        print(mw)
#        print(lw)
#        print(sess.run(grad,feed_dict={x:data}))
#        wanna_see(data,x_hat_1)
        writer.close()
        sess.close()
    tf.reset_default_graph()
    wanna_see(data,x_hat_1)
    return z_sample,x_hat_1