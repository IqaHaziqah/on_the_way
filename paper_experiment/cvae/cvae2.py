# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:05:53 2018

@author: zhouying
整体的模型架构参见本文件夹下的cvae.vsdx
先训练classifier，然后在cvae的训练过程中，不断加入classifier的反馈
"""

from __future__ import division
import sys
sys.path.append('..\\')
import tensorflow as tf
import numpy as np
from myutil2 import Dataset,compute_gradient_comparitive_error
tf.reset_default_graph()
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

'''共用的函数'''
def weight_variable(shape,ini):
    Hin,Hout = shape
    if ini =='lecun':
        b = np.sqrt(3.0/Hin)
        initial = np.random.uniform(low=-b,high=b,size=shape)
    elif ini =='msra':
        initial = np.random.normal(scale=2.0/Hin,size=shape)
    elif ini == 'xaiver':        
        b = 1.0*np.sqrt(6.0/(Hin+Hout))
        initial = np.random.uniform(low=-b,high=b,size=shape)
    elif ini =='he':
        b = np.sqrt(2.0/Hin)        
        initial = np.random.uniform(low=-b,high=b,size=shape)
    else:
        initial = np.random.normal(scale=0.001,size=shape)
    return initial


class CVAE(object):
    def __init__(self, sess,para_o):
        self.sess = sess
        #parameters for the network and training 
        self.epoch = para_o['epochs']
        self.batch_size = para_o['batch_size']
        self.z_dim = para_o['latent_dim']        # dimension of noise-vector
        self.c_dim = 1 
        self.learning_rate = para_o['learning_rate']
        self.r = 0.05
        self.beta1 = 0.5
        self.hidden1,self.hidden2,_,_ = para_o['hidden']
        self.lamda1,self.lamda2,self.lamda3,self.lamda4,self.lamda5 = para_o['lamda']
        self.input_dim = para_o['input_dim']
        #input
        self.image = tf.placeholder('float64',shape=[None,self.input_dim])
        self.label = tf.placeholder('float64',shape=[None,self.c_dim])
        self.rp = tf.placeholder('float64',shape=[None,self.input_dim])
        self.rn = tf.placeholder('float64',shape=[None,self.input_dim])
        self.z = tf.placeholder('float64')
        self.posr = 0
        self.negr = 0
        self.posp = 0
        self.negp = 0
        self.regularizer = tf.contrib.layers.l2_regularizer(0.01)
        self.ini = para_o['ini']
    
    
    def dense(self,inputs, shape, name, bn=False, act_fun=None):
        ini = weight_variable(shape,self.ini)
        W = tf.get_variable(name + ".w", initializer=ini)
        b = tf.get_variable(name + ".b", initializer=(tf.zeros((1, shape[-1])) + 0.1))
        y = tf.add(tf.matmul(inputs, W), b)
    
        def batch_normalization(inputs, out_size, name, axes=0):
            mean, var = tf.nn.moments(inputs, axes=[axes])
            scale = tf.get_variable(name=name + ".scale", initializer=tf.ones([out_size]))
            offset = tf.get_variable(name=name + ".shift", initializer=tf.zeros([out_size]))
            epsilon = 0.001
            return tf.nn.batch_normalization(inputs, mean, var, offset, scale, epsilon, name=name + ".bn")
    
        if bn:
            y = batch_normalization(y, shape[1], name=name + ".bn")
        if act_fun:
            y = act_fun(y)
        return y
    
    # Gaussian Encoder 高斯编码器
    def encoder(self,x,reuse=tf.AUTO_REUSE):
        with tf.variable_scope("encoder", reuse=reuse):
            net = self.dense(x,[self.input_dim,self.hidden],name='en_fc1',act_fun=tf.nn.relu)
            net =  tf.concat([net,self.label],1)           
            gaussian_params = self.dense(net,shape=[self.hidden1+1,self.z_dim*2],name='en_fc2')
            # The mean parameter is unconstrained 将输出分为两块，z_mean和z_log_var,就是高斯分布的均值和标准差
            # 分出的前（64,62）代表均值， 后（64,62）处理后代表标准差
            
            mean = gaussian_params[:,:self.z_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            # 标准差必须是正值。 用softplus参数化并为数值稳定性添加一个小的epsilon, softplus为y=log(1+ex)
            stddev = tf.nn.softplus(gaussian_params[:,self.z_dim:])
        return mean, stddev

    def generator(self,label, z, reuse=tf.AUTO_REUSE):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.concat([z,self.label],1)
            net = self.dense(net,[self.hidden1,self.hidden2],name='ge_fc1',act_fun=tf.nn.relu)
            net = self.dense(net,[self.hidden2,self.hidden1],name='ge_fc2',act_fun=tf.nn.relu)
            net = self.dense(net,[self.hidden2,self.input_dim],naem='ge_fc3')
        return net

    def train_CLA(self,args):
        data,label = args
        label = np.expand_dims(label,1)
        data = np.concatenate([data,label],1)
        self.num_batches = int(data.shape[0]/self.batch_size)
        data = Dataset(data)
        for epoch in range(self.epoch):
            for idx in range(self.num_batches):
                batch_images= data.next_batch(self.batch_size)[0]
                batch_labels = np.expand_dims(batch_images[:,-1],1)
                batch_images = batch_images[:,:-1]
                self.sess.run(self.cla_optim,feed_dict={self.image:batch_images,self.label:batch_labels})    
        return


    def build_model(self):
        """ Loss Function """
        
        #classifier
       
        
        #定义classifier的参数
        ini = weight_variable([self.input_dim,1],self.ini)
        self.w1 = tf.Variable('c1_w', initializer=ini)
        self.b1 = tf.Variable('c1_b', initializer=(tf.zeros((1, 1) + 0.1)))
        self.y1 = tf.nn.sigmoid(tf.add(tf.matmul(self.image,self.w1),self.b1))
        self.loss1 = tf.reduce_mean(-tf.reduce_sum(self.label*tf.log(self.y1),reduction_indices=1))
        
        ''' encoding，直接返回 mu 和 sigma'''
        self.mu, self.sigma = self.encoder(x=self.image)
        '''真实的z'''
        z = self.mu + self.sigma * np.random.normal(size=self.z_dim)
        KL = tf.reduce_mean(-0.5 * tf.reduce_sum(
            tf.square(self.mu) + tf.square(self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1,
            [1]))
        
#        KL = tf.reduce_mean(-0.5*tf.reduce_sum(1+self.logvar-tf.square(self.mu)-tf.exp(self.logvar),axis=-1))
        '''xf为原始z生成的样本'''
        xf = self.generator(label=self.label,z=z)
        

        #生成时需要用到的op
        self.xp = self.generator(self.label,self.z)
        
        
        self.sess.run(tf.assign(self.w2,self.w1))
        self.sess.run(tf.assign(self.b2,self.b1))
        self.y2 = tf.nn.sigmoid(tf.add(tf.matmul(self.image,self.w2),self.b2))
        self.loss2 = tf.reduce_mean(-tf.reduce_sum(self.label*tf.log(self.y2),reduction_indices=1))
        t_vars = tf.trainable_variables()
        c2_vars = [var for var in t_vars if 'c2_' in var.name]
        

        '''接下来计算LG和LE即可，其中难处在于如何计算feature center'''
        '''认为feature center为net'''
        
        '''这部分为classifier的损失函数'''
        _,_,xcr = self.classifier(self.image,name='classifier1')
        _,_,xcf = self.classifier(xf,name='classifier1')


        LG = 0.5*(tf.reduce_mean(tf.square(self.image-xf)))#+0.1*tf.reduce_mean(tf.square(dxr[2]-dxf[2]))+0.1*tf.reduce_mean(tf.square(xcr-xcf)))

        """ Training """
        # divide trainable variables into a group for D and a group for G

        
#        print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.dec_loss = self.lamda2*LG  #+tf.reduce_mean(regularization_loss)
        self.enc_loss = self.lamda1*KL + self.lamda2*LG       # +tf.reduce_mean(regularization_loss)
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.update_cla = tf.train.AdamOptimizer(self.learning_rate)\
                .minize(self.loss2,var_list=c2_vars)
        
        dec_vars = [var for var in t_vars if 'ge_' in var.name]
        enc_vars = [var for var in t_vars if 'en_' in var.name]
        c1_vars = [var for var in t_vars if 'c1_' in var.name]
        self.sess.run(self.update_cla,feed_dict={self.image:self.xp,self.label:np.ones(self.batch_size),self.z:np.random.normal([self.batch_size,self.z_dim])})
        
        
        
        
        
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            
            self.cla_optim = tf.train.AdamOptimizer(self.learning_rate) \
                .minimize(self.c_loss, var_list=c1_vars)            
            self.dec_optim = tf.train.AdamOptimizer(self.learning_rate) \
                      .minimize(self.dec_loss, var_list=dec_vars)
            self.enc_optim = tf.train.AdamOptimizer(self.learning_rate) \
                .minimize(self.enc_loss, var_list=enc_vars)

        self.sess.run(tf.global_variables_initializer())
           

    def train_VAE(self,args):
        data,label = args
        # initialize all variables
#        data = np.concatenate((data,label),axis=1)
        pos = data[label==1]
        neg = data[label==0]
        # graph inputs for visualize training results
        # 创建高斯分布z，均值为0，方差为1
        pos = Dataset(pos)
        neg = Dataset(neg)
        self.num_batches = int(data.shape[0]/self.batch_size)
        # loop for epoch
        counter = 0
        for epoch in range(self.epoch):
            
            # get batch data
            for idx in range(0, self.num_batches):
                n = int(self.batch_size/2)
                batch_images = np.concatenate((pos.next_batch(n)[0],neg.next_batch(n)[0]),0)
#                print(batch_images.shape)
                batch_labels = np.concatenate((np.ones([n,1]),np.zeros([n,1])))
#                print(batch_labels)
#                print(batch_images.shape,batch_labels.shape)
                # 创建高斯分布z，均值为0，方差为1
                batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
#                print(batch_z.shape)
#                batch_z = tf.cast(batch_z,tf.float32)
                feed = {self.image:batch_images,self.rn:batch_images[np.squeeze(batch_labels)==0],
                                                              self.rp:batch_images[np.squeeze(batch_labels)==1],
                                                   self.z: batch_z,self.label:batch_labels}

                # update D network sess.run喂入数据优化更新D网络
#                if epoch == self.epoch-2:
#                    self.graident_check(feed)
                
                # update decoder network
                ff = self.sess.run(self.g,feed_dict=feed)
                _,dec_loss = self.sess.run([self.dec_optim,self.dec_loss],
                                                   feed_dict=feed)

                # update encoder network
                _, enc_loss = self.sess.run([self.enc_optim, self.enc_loss],
                                                        feed_dict=feed)
                
#                print(xcp)
#                print(enc_loss,dec_loss,d_loss,c_loss,lgc,lgd)
                # display training status
                counter += 1
                if counter%5 == 0:
#                    print("Epoch:[%2d] d_loss: %.5f, gen_loss: %.5f, enc_loss: %.5f, cla_loss: %.5f" \
#                          % (epoch, d_loss, dec_loss, enc_loss,c_loss))
#                    print(ff[0])
                    pass

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
    
    def graident_check(self,feed):
        t_vars = tf.trainable_variables()
        dec_vars = [var for var in t_vars if 'ge_' in var.name]
        enc_vars = [var for var in t_vars if 'en_' in var.name]
        c_vars = [var for var in t_vars if 'c_' in var.name]
        with self.sess:
            r1 = compute_gradient_comparitive_error(x=d_vars[0],x_shape=[self.input_dim,self.batch_size],y=self.d_loss,y_shape=[1],extra_feed_dict=feed)
            r2 = compute_gradient_comparitive_error(x=dec_vars[0],x_shape=[self.z_dim+1,self.hidden2],y=self.dec_loss,y_shape=[1],extra_feed_dict=feed)
            r3 = compute_gradient_comparitive_error(x=enc_vars[0],x_shape=[self.input_dim,self.batch_size],y=self.enc_loss,y_shape=[1],extra_feed_dict=feed)
            r4 = compute_gradient_comparitive_error(x=c_vars[0],x_shape=[self.input_dim,self.batch_size],y=self.c_loss,y_shape=[1],extra_feed_dict=feed)
        print(r2,r3,r4)
        print(r1)
        return 
    
    def oversampling(self,ovsize):
        batch_z = np.random.normal(size=[ovsize,self.z_dim])
        cp = np.ones([ovsize,1])
        ge = self.sess.run(self.xp,feed_dict={self.label:cp,self.z:batch_z})
        return ge
    def close(self):
        self.sess.close()

import scipy
para_o = {
    'latent_dim':2,
    'epochs':15,
    'batch_size':100, #为偶数
    'learning_rate':0.001,#两个分类型网络的学习率
#    'PRE':sklearn.preprocessing.StandardScaler,
    'hidden':[20,10,10,10],   #分别是指编码器中的hidden1，hidden2，分类器中的hidden1，hidden2
    'lamda':[2,1,1,1,0],
    'dataset_name':'ionosphere',
    'ini':'msra'
        }
mydata = scipy.io.loadmat('..\\MNIST_data\\UCI\\'+para_o['dataset_name']+'.mat')
#for linux
#mydata = scipy.io.loadmat('../MNIST_data/UCI/ionosphere.mat')
data = np.array(mydata['data'])
label = np.squeeze(mydata['label'])#np.transpose

para_o['input_dim'] = data.shape[1]
sess = tf.Session()
c = CVAE(sess,para_o)
c.build_model()
c.train_CLA([data,label])
c.train_VAE([data,label])
#ans = c.oversampling(317)