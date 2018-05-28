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
from keras import backend as K 
from myutil2 import readdata,Dataset
tf.reset_default_graph()


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
        #parameters for the network and training 
        self.epoch = para_o['epochs']
        self.batch_size = para_o['batch_size']
        self.z_dim = para_o['latent_dim']        # dimension of noise-vector
        self.c_dim = 1 
        self.learning_rate = para_o['learning_rate']
        self.r = 0.05
        self.beta1 = 0.5
        self.hidden1,self.hidden2,self.hidden11,self.hidden12 = para_o['hidden']
        self.lamda1,self.lamda2,self.lamda3,self.lamda4 = para_o['lamda']
        self.input_dim = para_o['input_dim']
        #input
        self.image = tf.placeholder('float32',shape=[None,self.input_dim])
        self.label = tf.placeholder('float32',shape=[None,self.c_dim])
        self.z = tf.placeholder('float32')
        self.posr = 0
        self.negr = 0
        self.posp = 0
        self.negp = 0
        
    def linear(self,x,shape,scope):
        w = tf.get_variable(initializer=tf.random_normal(shape,stddev=np.sqrt(2/shape[0])),name=scope+'_w')
        b = tf.get_variable(initializer=tf.ones(shape[1])*0.1,name=scope+'_b')
        return tf.matmul(x,w)+b
    
    # Gaussian Encoder 高斯编码器
    def encoder(self,x,reuse=tf.AUTO_REUSE):
        with tf.variable_scope("encoder", reuse=reuse):
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

    def generator(self,label, z, reuse=tf.AUTO_REUSE):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.concat([z,self.label],1)
            net = tf.nn.relu((self.linear(net, shape=[self.z_dim+1,self.hidden2], scope='ge_fc1')))
            net = tf.nn.relu((self.linear(net, shape=[self.hidden2,self.hidden1], scope='ge_fc2')))
            net = self.linear(net, shape=[self.hidden1,self.input_dim], scope='ge_fc3')
        return net

    def discriminator(self, x, reuse=tf.AUTO_REUSE):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            net = tf.nn.relu(self.linear(x,shape=[self.input_dim,self.hidden1],scope='d_fc1'))
            net = tf.nn.relu(self.linear(net,shape=[self.hidden1,self.hidden2],scope='d_fc2')) 
            out_logit = self.linear(net,shape=[self.hidden2,1], scope='d_fc3')
            out = tf.nn.sigmoid(out_logit)
        return out, out_logit, net
    
    def classifier(self,x,reuse=tf.AUTO_REUSE):
        with tf.variable_scope('classifier',reuse=reuse):
            net = tf.nn.relu(self.linear(x,shape=[self.input_dim,self.hidden1],scope='c_fc1'))
            net = tf.nn.relu(self.linear(net,shape=[self.hidden1,self.hidden2],scope='c_fc2')) 
            out_logit = self.linear(net,shape=[self.hidden2,1], scope='c_fc3')
            out = tf.nn.sigmoid(out_logit)
        return out, out_logit, net
  
    def feature_center(self,logits,feature):
        #return mean of positive feature center and the negative feature center
        label = tf.cast(logits,tf.int32)
#        with self.sess.as_default():
#            print(label.eval())
        pos = []
        neg = []
        
        with self.sess.as_default():
            for i in range(self.batch_size):
                
                if label[i]==0:
                    neg.append(tf.slice(feature,begin=[i,0],size=[0,-1]))
                else:
                    pos.append(tf.slice(feature,begin=[i,0],size=[0,-1]))
#        tf.Print(neg,neg)
#        tf.Print(pos,pos)
#        print(tf.reduce_mean(pos,axis=0),tf.reduce_mean(neg,axis=0))
        print('yes')
#        print(tf.reduce_mean(pos),tf.reduce_mean(neg))
        return tf.reduce_mean(pos),tf.reduce_mean(neg)

    def build_model(self):
        """ Loss Function """
        
        #classifier
        self.y_pred = self.classifier(x=self.image)
#        LC = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=self.y_pred))
        LC = tf.reduce_mean(-tf.reduce_sum(self.label*tf.log(self.y_pred[0]),reduction_indices=1))
        ''' encoding，直接返回 mu 和 sigma'''
        self.mu, self.sigma = self.encoder(x=self.image)
        '''真实的z'''
        z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        KL = tf.reduce_mean(-0.5 * tf.reduce_sum(
            tf.square(self.mu) + tf.square(self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1,
            [1]))
        '''xf为原始z生成的样本'''
        xf = self.generator(label=self.label,z=z)
        zp = self.z
        cp = np.random.randint(0,2,size=[self.batch_size])
        '''xp为采样后生成的样本'''
        '''这部分为分辨器中的结果和损失函数'''
        xp = self.generator(cp,zp)
        dxr= self.discriminator(x=self.image)
        dxf= self.discriminator(x=xf)
        dxp= self.discriminator(x=xp)
        y_pos = K.ones_like(self.label)
        y_neg = K.zeros_like(self.label)
#        loss_real = tf.nn.softmax_cross_entropy_with_logits(lables=y_pos,logits=dxr)
        loss_real = -tf.reduce_sum(y_pos*tf.log(1e-8+dxr[0]))
#        loss_fake_f = tf.nn.softmax_cross_entropy_with_logits(labels=y_neg,logits=dxf)
        loss_fake_f = -tf.reduce_sum(y_neg*tf.log(1e-8+dxf[0]))
#        loss_fake_p = tf.nn.softmax_cross_entropy_with_logits(labels=y_neg,logits=dxp)
        loss_fake_p = -tf.reduce_sum(y_neg*tf.log(1e-8+dxp[0]))
        
#        loss_fake_p = tf.reduce_sum(tf.square(dxp[0]-y_neg))
        LD = tf.reduce_mean(loss_real+loss_fake_f+loss_fake_p)
        

        #生成时需要用到的op
        self.xp = self.generator(self.label,self.z)
        '''接下来计算LG和LE即可，其中难处在于如何计算feature center'''
        '''认为feature center为net'''

        
        _,_,ndxr = self.discriminator(self.image)
        _,_,ndxp = self.discriminator(xp)
        self.LGD = 0.5*tf.square(tf.reduce_mean(ndxr)-tf.reduce_mean(ndxp))
        '''这部分为classifier的损失函数'''
#        fee = self.image[self.label==0,:]
        
        _,_,xcp = self.classifier(xp)        
        _,_,xcr = self.classifier(self.image)
        _,_,xcf = self.classifier(xf)
#        LGC = 0.5*(tf.reduce_mean(tf.square(cr0-cp0))+tf.reduce_mean(tf.square(cr1-cp1)))
        
        x = tf.constant(xcp)
        with tf.Session() as sess:
            print (sess.run(x))
        
        ans = self.feature_center(cp,xcp)
        
        print(ans)
        self.posp = (self.posp+ans[0])/2
        self.negp = (self.negp+ans[1])/2
        ans = self.feature_center(self.label,xcr)
        self.posr = (self.posr+ans[0])/2
        self.negr = (self.negr+ans[1])/2
        self.LGC = 0.5*(tf.square(self.posp-self.posr)+tf.square(self.negp-self.negr))

        LG = 0.5*(tf.reduce_mean(tf.square(self.image-xp))+tf.reduce_mean(tf.square(dxr[2]-dxf[2]))+tf.reduce_mean(tf.square(xcr[2]-xcf[2])))
      
        self.d_loss = LD
        self.dec_loss = self.lamda2*LG + self.lamda3*self.LGD# + self.lamda4*self.LGC
        self.c_loss = LC
        self.enc_loss = self.lamda1*KL + self.lamda2*LG
        
        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        dec_vars = [var for var in t_vars if 'ge_' in var.name]
        enc_vars = [var for var in t_vars if 'en_' in var.name]
        c_vars = [var for var in t_vars if 'c_' in var.name]

        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            
            self.cla_optim = tf.train.AdamOptimizer(self.learning_rate) \
                .minimize(self.c_loss, var_list=c_vars)
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate*5) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.dec_optim = tf.train.AdamOptimizer(self.learning_rate) \
                      .minimize(self.dec_loss, var_list=dec_vars)
            self.enc_optim = tf.train.AdamOptimizer(self.learning_rate*5) \
                .minimize(self.enc_loss, var_list=enc_vars)
            

    def train(self,name):
        data = readdata(name,withlabel=True)
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # graph inputs for visualize training results
        # 创建高斯分布z，均值为0，方差为1
        data = Dataset(data)
        self.num_batches = int(data._num_examples/self.batch_size)
        # loop for epoch
        start_time = time.time()
        counter = 0
        for epoch in range(self.epoch):
            
            # get batch data
            for idx in range(0, self.num_batches):
                batch_images = data.next_batch(self.batch_size)[0]
                batch_labels = batch_images[:,-1:]
#                print(batch_labels)
#                print(batch_images.shape,batch_labels.shape)
                # 创建高斯分布z，均值为0，方差为1
                batch_z = np.random.normal(size=[self.batch_size, self.z_dim])
#                print(batch_z.shape)
#                batch_z = tf.cast(batch_z,tf.float32)
                

                # update D network sess.run喂入数据优化更新D网络
                _,  d_loss = self.sess.run([self.d_optim,  self.d_loss],
                                        feed_dict={self.image:batch_images[:,:-1],self.z: batch_z,self.label:batch_labels})
                
                # update decoder network
                _,dec_loss,lgc,lgd = self.sess.run([self.dec_optim,self.dec_loss,self.LGC,self.LGD],feed_dict={self.image:batch_images[:,:-1], self.z: batch_z,self.label:batch_labels})

                # update encoder network
                _, enc_loss = self.sess.run([self.enc_optim, self.enc_loss],
                                                        feed_dict={self.image:batch_images[:,:-1], self.z: batch_z,self.label:batch_labels})

                _,c_loss = self.sess.run([self.cla_optim,self.c_loss],feed_dict={self.image:batch_images[:,:-1], self.z: batch_z,self.label:batch_labels})
                print(d_loss,dec_loss,enc_loss,c_loss,lgc,lgd)
                # display training status
#                counter += 1
#                if np.mod(counter, 50) == 0:
#                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, dec_loss: %.8f, enc_loss: %.8f" \
#                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, dec_loss, enc_loss))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model

    def oversampling(self,ovsize):
        batch_z = np.random.normal(size=[ovsize,self.z_dim])
        cp = np.random.randint(0,1,size=[ovsize,1])
        ge = self.sess.run(self.xp,feed_dict={self.label:cp,self.z:batch_z})
        return ge

para_o = {
    'hidden_encoder_dim':15,                                    
    'hidden_decoder_dim':15, 
    'latent_dim':2,
    'lam':0,
    'epochs':1,
    'batch_size':350,
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
    'lamda':[3,1,0.001,0.001],
    'dataset_name':'ionosphere',
    'input_dim':34
        }

sess = tf.Session()
c = VAE_GAN(sess,para_o)
c.build_model()
c.train('ionosphere')
#print(c.oversampling(10))
