# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:26:06 2018

@author: zhouying
"""

from __future__ import division
import sys
sys.path.append('..\\')
import time
import tensorflow as tf
import numpy as np
import keras 
from keras import backend as K 
tf.reset_default_graph()
para_o = {
    'hidden_encoder_dim':15,                                    
    'hidden_decoder_dim':15, 
    'latent_dim':2,
    'lam':0,
    'epochs':120,
    'batch_size':10,
    'learning_rate':0.0002,
    'ran_walk':False,
    'check':False,
    'trade_off':0.5,
    'activation':tf.nn.relu,
    'optimizer':tf.train.AdamOptimizer,
    'norm':True,
    'decay':0.9,
    'initial':3,
#    'PRE':sklearn.preprocessing.StandardScaler,
    'dataset':'breastw',
    'hidden':[20,10,20,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda1':3,
    'lamda2':1,
    'lamda3':0.001,
    'lamda4':0.001,
    'dataset_name':'ionosphere'
        }

def load_mnist(name):
    from myutil2 import Dataset
    import scipy.io as scio
    filepath = '..\\MNIST_data\\UCI\\'+name+'.mat'
    mydata = scio.loadmat(filepath)
    data = np.array(mydata['data'])
    label = np.array(mydata['label'])
    data = np.concatenate((data,label),axis=1)
    data = Dataset(data)
    return data


class VAE_GAN(object):
    def __init__(self, sess,para_o):
        self.sess = sess
        #self.test_dir = test_dir
        self.epoch = para_o['epochs']
        self.batch_size = para_o['batch_size']
        self.z_dim = para_o['latent_dim']        # dimension of noise-vector
        self.c_dim = 1
        self.dataset_name = para_o['dataset']
        # train
        self.learning_rate = para_o['learning_rate']
        self.r = 0.05
        self.beta1 = 0.5
        self.hidden1,self.hidden2,self.hidden11,self.hidden12 = para_o['hidden']
        self.lamda1 = para_o['lamda1']
        self.lamda2 = para_o['lamda2']
        self.lamda3 = para_o['lamda3']
        self.lamda4 = para_o['lamda4']
        # test
        # load mnist
        self.data_X = load_mnist(self.dataset_name)
        self.input_dim = self.data_X.num_features()
        # get number of batches for a single epoch
        self.num_batches = self.data_X.num_examples() // self.batch_size
#        else:
#            raise NotImplementedError
        
    def linear(self,x,shape,scope):
        w = tf.get_variable(initializer=tf.random_normal(shape,stddev=np.sqrt(2/shape[0])),name=scope+'_w')
        b = tf.get_variable(initializer=tf.ones(shape[1])*0.1,name=scope+'_b')
        return tf.matmul(x,w)+b
    
    # Gaussian Encoder 高斯编码器
    def encoder(self, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            x = self.image
            net = tf.nn.relu(self.linear(x,shape=[self.input_dim,self.hidden1],scope='en_fc1'))
            net = tf.nn.relu(self.linear(net,shape=[self.hidden1,self.hidden2],scope='en_fc2'))
            net =  tf.concat([net,self.label],1)           
            gaussian_params = self.linear(net, shape=[self.hidden2+1,2 * self.z_dim], scope='en_fc3')
            # The mean parameter is unconstrained 将输出分为两块，z_mean和z_log_var,就是高斯分布的均值和标准差
            # 分出的前（64,62）代表均值， 后（64,62）处理后代表标准差
            mean = gaussian_params[:,:self.z_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            # 标准差必须是正值。 用softplus参数化并为数值稳定性添加一个小的epsilon, softplus为y=log(1+ex)
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:,self.z_dim:])
        return mean, stddev

    def generator(self,label, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.concat([z,self.label],1)
            net = tf.nn.relu((self.linear(net, shape=[self.z_dim+1,self.hidden2], scope='ge_fc1')))
            net = tf.nn.relu((self.linear(net, shape=[self.hidden2,self.hidden1], scope='ge_fc2')))
            net = self.linear(net, shape=[self.hidden1,self.input_dim], scope='ge_fc3')
        return net

    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            net = tf.nn.relu(self.linear(x,shape=[self.input_dim,self.hidden1],scope='d_fc1'))
            net = tf.nn.relu(self.linear(net,shape=[self.hidden1,self.hidden2],scope='d_fc2')) 
            out_logit = self.linear(net,shape=[self.hidden2,1], scope='d_fc3')
            out = tf.nn.sigmoid(out_logit)
        return out, out_logit, net
    
    def classifier(self,x,reuse=False):
        with tf.variable_scope('classifier',reuse=reuse):
            net = tf.nn.relu(self.linear(x,shape=[self.input_dim,self.hidden1],scope='c_fc1'))
            net = tf.nn.relu(self.linear(net,shape=[self.hidden1,self.hidden2],scope='c_fc2')) 
            out_logit = self.linear(net,shape=[self.hidden2,1], scope='c_fc3')
            out = tf.nn.sigmoid(out_logit)
        return out, out_logit, net
    
    def NLLNormal(self, pred, target):
        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
        tmp *= -multiplier
        tmp += c
        return tmp

    def build_model(self):
        # some parameters

        """ Graph Input """
        # images,label
        self.image = tf.placeholder(tf.float32,shape=[None,self.input_dim])
        self.label = tf.placeholder(tf.float32,shape=[None,1])
        
        # noises
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        """ Loss Function """
        #classifier
        self.y_pred = self.classifier(x=self.image,reuse=False)
#        LC = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=self.y_pred))
        LC = tf.reduce_mean(-tf.reduce_sum(self.y_pred[0]*tf.log(self.label),reduction_indices=1))
        ''' encoding，直接返回 mu 和 sigma'''
        self.mu, sigma = self.encoder(is_training=True, reuse=False)
        '''真实的z'''
        z = self.mu + sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        KL = tf.reduce_mean(-0.5 * tf.reduce_sum(
            tf.square(self.mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1,
            [1]))
        '''xf为原始z生成的样本'''
        xf = self.generator(label=self.label,z=z)
        zp = tf.random_normal(shape=[self.batch_size,self.z_dim],dtype=tf.float32)
        cp = np.random.randint(0,1,size=[self.batch_size])
        '''xp为采样后生成的样本'''
        '''这部分为分辨器中的结果和损失函数'''
        xp = self.generator(cp,zp,reuse=True)
        dxr= self.discriminator(x=self.image)
        dxf= self.discriminator(x=xf,reuse=True)
        dxp= self.discriminator(x=xp,reuse=True)
        y_pos = K.ones_like(self.label)
        y_neg = K.zeros_like(self.label)
#        loss_real = tf.nn.softmax_cross_entropy_with_logits(lables=y_pos,logits=dxr)
        loss_real = -tf.reduce_sum(dxr[0]*tf.log(y_pos))
#        loss_fake_f = tf.nn.softmax_cross_entropy_with_logits(labels=y_neg,logits=dxf)
        loss_fake_f = -tf.reduce_sum(dxf[0]*tf.log(y_neg))
#        loss_fake_p = tf.nn.softmax_cross_entropy_with_logits(labels=y_neg,logits=dxp)
        loss_fake_p = -tf.reduce_sum(dxp[0]*tf.log(y_neg))
        LD = K.mean(loss_real+loss_fake_f+loss_fake_p)
        
        self.xp = self.generator(self.label,self.z,reuse=True)
        '''接下来计算LG和LE即可，其中难处在于如何计算feature center'''
        '''认为feature center为net'''

        
        _,_,ndxr = self.discriminator(self.image,reuse=True)
        _,_,ndxp = self.discriminator(xp,reuse=True)
        LGD = 0.5*tf.square(tf.reduce_mean(ndxr)-tf.reduce_mean(ndxp))
        '''这部分为classifier的损失函数'''
        _,_,cr0 = self.classifier(self.image[self.label==0,:],reuse=True)
        _,_,cr1 = self.classifier(self.image[self.label==1,:],reuse=True)
        _,_,cp0 = self.classifier(xp[cp==0,:],reuse=True)
        _,_,cp1 = self.classifier(xp[cp==1,:],reuse=True)
        xcr = self.classifier(self.image,reuse=True)
        xcf = self.classifier(xf,reuse=True)
        LGC = 0.5*(tf.reduce_mean(tf.square(cr0-cp0))+tf.reduce_mean(tf.square(cr1-cp1)))
        LG = 0.5*(tf.reduce_mean(tf.square(self.image-xp))+tf.square(dxr-dxf)+tf.square(xcr-xcf))
      
        self.d_loss = LD
        self.dec_loss = self.lamda2*LG + self.lamda3*LGD + self.lamda4*LGC
        self.c_loss = LC
        self.enc_loss = self.lamda1*KL + self.lamda2*LG
        
        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        dec_vars = [var for var in t_vars if 'de_' in var.name]
        enc_vars = [var for var in t_vars if 'en_' in var.name]
        c_vars = [var for var in t_vars if 'c_' in var.name]

        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            
            self.cla_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.c_loss, var_list=c_vars)
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.dec_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.dec_loss, var_list=dec_vars)
            self.enc_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                .minimize(self.enc_loss, var_list=enc_vars)
            

    def train(self):

        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # graph inputs for visualize training results
        # 创建高斯分布z，均值为0，方差为1
        self.sample_z = np.random.normal(self.batch_size, self.z_dim)

        # loop for epoch
        start_time = time.time()
        counter = 0
        for epoch in range(0, self.epoch):
            
            # get batch data
            for idx in range(0, self.num_batches):
                batch_images = self.data_X.next_batch(self.batch_size)
                # 创建高斯分布z，均值为0，方差为1
                batch_z = np.random.normal(self.batch_size, self.z_dim)
                

                # update D network sess.run喂入数据优化更新D网络
                _,  d_loss = self.sess.run([self.d_optim,  self.d_loss],
                                        feed_dict={self.inputs:batch_images[:,:-1], self.z: batch_z})

                # update decoder network
                self.sess.run([self.dec_optim],feed_dict={self.inputs: batch_images[:,:-1], self.z: batch_z})

                # update encoder network
                _, enc_loss = self.sess.run([self.enc_optim, self.enc_loss],
                                                        feed_dict={self.inputs: batch_images[:,:-1], self.z: batch_z})

                self.sess.run([self.cla_optim],feed_dict={self.inputs:batch_images[:,:-1]})

                # display training status
                counter += 1
#                if np.mod(counter, 50) == 0:
#                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, dec_loss: %.8f, enc_loss: %.8f" \
#                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, dec_loss, enc_loss))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model

    def oversampling(self,ovsize):
        batch_z = np.random_normal(ovsize)
        cp = np.randint((0,1),ovsize)
        ge = self.sess.run(self.xp,feed_dict={self.label:cp,self.z:batch_z})
        return ge

sess = tf.Session()
c = VAE_GAN(sess,para_o)
c.build_model()
c.train()
c.oversampling(10)