#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:38:51 2017

@author: zhouying
"""

def mnist_vae(data,gene_size):
    import tensorflow as tf
    from myutil import Dataset
    import numpy as np
    from scipy.stats import norm
    from myutil import random_walk
    mnist = Dataset(data)
    input_dim = data.shape[1]
    hidden_encoder_dim = 18
    hidden_decoder_dim = 18 
    latent_dim = 5
    lam = 0
    epochs = 50
    batch_size = 2
    learning_rate = 0.001
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial)

    x = tf.placeholder("float", shape=[None, input_dim])
    input_z = tf.placeholder("float",shape = [None,latent_dim])
    l2_loss = tf.constant(0.0)
    
    W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim])
    b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
    l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)
#    variable_summaries(W_encoder_input_hidden, 'W_encoder_input_hidden')
    
    # Hidden layer encoder
    hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)
    
    W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
    b_encoder_hidden_mu = bias_variable([latent_dim])
    l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)
    
    # Mu encoder
    mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

    W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
    b_encoder_hidden_logvar = bias_variable([latent_dim])
    l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

# Sigma encoder
    logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
    epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable
    std_encoder = tf.exp(0.5 * logvar_encoder)
    z = mu_encoder + tf.multiply(std_encoder, epsilon)

    W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
    b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
    l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
    hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

    W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
    b_decoder_hidden_reconstruction = bias_variable([input_dim])
    l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

    KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)
#    KLD = 0
    kld = tf.reduce_mean(KLD)
    x_hat = (tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction)
#    BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x), reduction_indices=1)
#    BCE = tf.reduce_sum(tf.abs(x_hat-x))
    BCE = tf.reduce_sum(tf.pow(x_hat-x,2))
    loss = tf.reduce_mean(BCE + KLD)

    regularized_loss = loss + lam * l2_loss

#    loss_summ = tf.summary.scalar("lowerbound", loss)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(regularized_loss)
    
    hidden_decoder_1 = tf.nn.relu(tf.matmul(input_z, W_decoder_z_hidden) + b_decoder_z_hidden)
    x_hat_1 = (tf.matmul(hidden_decoder_1, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction)
    
    with tf.Session() as sess:
        total = int(data.shape[0]/batch_size)
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            for j in range(total):
                _, cur_loss,cur_kld = sess.run([train_step,loss,kld], feed_dict={x: mnist.next_batch(batch_size)[0]})
#                if j % 5 == 0:
#                    print("Step {0} | Loss: {1}".format((i*total+j), cur_loss))
#                    print("Step {0} | kld: {1}".format((i*total+j), cur_kld))
        zz = sess.run([z],feed_dict={x:data})
#        print(zz.shape)
        z_sample = random_walk(zz,gene_size)
#        z_sample = np.random.randn(gene_size,latent_dim)
        x_hat_1 = sess.run([x_hat_1],feed_dict = {input_z:z_sample})
        
#        z_sample = np.random.randn(gene_size,latent_dim)
#        x_hat_1_1 = sess.run([x_hat_1],feed_dict = {input_z:z_sample})
#    return {x_hat_1[0],x_hat_1_1[0]}
    return x_hat_1[0]
#    return x_train_generator