# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:36:41 2018

@author: zhouying
"""
import sys
sys.path.append('..\\')
import tensorflow as tf


tf.reset_default_graph()
class V(object):
    def __init__(self,sess):
        self.a = tf.placeholder(tf.float32)
        self.b = tf.placeholder(tf.float32)
        self.sess = sess
        return
    
    def build_model(self):
        a = self.a * self.b
        self.sum = a
        return 

    def train(self,a,b):
        c = self.sess.run(self.sum,feed_dict={self.a:a,self.b:b})
        print(c)
        return
    
sess = tf.Session()
c = V(sess)
c.build_model()
#a = c.sess.run(c.sum,feed_dict={c.a:1,c.b:2})
#print(a)
c.train(a=1,b=2)


#class GAN(object):
#    def __init__(self,sess,para_o):
#        self.hidden1,self.hidden2 = para_o['hidden']
#        self.learning_rate = para_o['learning_rate']
#        self.z_dim = para_o['latent_dim']
#        self.x_dim = para_o['input_dim']
#        self.sess = sess
#        self.batch_size = para_o['batch_size']
#        self.x_real = tf.placeholder('float32',shape=[None,self.x_dim])
#        self.z = tf.placeholder('float32',shape=[None,self.z_dim])
#        self.epoch = para_o['epochs']
#        
#    def add_layer(self,x,name,shape):
#        #基础层
#        w = tf.get_variable(name=name+'_w',initializer=tf.random_normal(shape=shape))
#        b = tf.get_variable(name=name+'_b',initializer=tf.ones(shape[1])*0.1)
#        return tf.matmul(x,w)+b
#    
#    
#    def generator(self,z,reuse):
#        #由z生成X_fake
#        with tf.variable_scope('generator',reuse=reuse):
#            net = tf.nn.relu(self.add_layer(x=z,shape=[self.z_dim,self.hidden2],name='ge_fc1'))
#            net = (self.add_layer(x=net,shape=[self.hidden2,self.hidden1],name='ge_fc2'))
#        return net
#    
#    def discrimitor(self,x,reuse):
#        #分辨器，分辨真实和产生的样本
#        with tf.variable_scope('discrimitor',reuse=reuse):
#            net = tf.nn.relu(self.add_layer(x=x,shape=[self.x_dim,self.hidden1],name='di_fc1'))
#            net = tf.nn.sigmoid(self.add_layer(x=net,shape=[self.hidden1,1],name='di_fc2'))
#        return net
#    
#    def bulid_model(self):
#        #将生成器和分辨器连接起来，并定义好各自的优化函数和优化变量列表
#        
#        self.x_fake = self.generator(self.z,reuse=tf.AUTO_REUSE)
#        y_real = tf.ones(shape=self.batch_size)
#        y_fake = tf.zeros(shape=self.batch_size)
#        y_fake_pre = self.discrimitor(self.x_fake,reuse=tf.AUTO_REUSE)
#        y_real_pre = self.discrimitor(self.x_real,reuse=tf.AUTO_REUSE)
#        self.lossd = tf.reduce_mean(tf.reduce_sum(tf.square(y_real-y_real_pre)))+tf.reduce_mean(tf.reduce_sum(tf.square(y_fake-y_fake_pre)))
#        self.lossg = tf.reduce_mean(tf.reduce_sum(tf.square(y_real-y_fake_pre)))
#        t_vars = tf.trainable_variables()
#        d_vars = [var for var in t_vars if 'discri' in var.name]
#        g_vars = [var for var in t_vars if 'generator' in var.name]
#        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#            self.optd = tf.train.AdamOptimizer(self.learning_rate).minimize(self.lossd,var_list=d_vars)
#            self.optg = tf.train.AdamOptimizer(self.learning_rate).minimize(self.lossg,var_list=g_vars)
#    
#    def train(self,name):
#        ima = readdata(name)
#        x = Dataset(ima[0])
#        z = np.random.normal(size=[self.batch_size,self.z_dim])
#        self.sess.run(tf.global_variables_initializer())
#        total = self.epoch*int(x._num_examples/self.batch_size)
#        for i in range(total):
#            batch = np.array(x.next_batch(self.batch_size))[0]
##            self.sess.run([self.lossd],feed_dict={self.x_real:batch})
#            self.sess.run([self.lossg],feed_dict={self.z:z,self.x_real:batch})
#
#para_o ={
#        'learning_rate':0.0001,
#        'latent_dim':2,
#        'input_dim':34,
#        'batch_size':10,
#        'epochs':100,
#        'hidden':[34,10]
#        }
##ans = readdata('ionosphere')
#sess = tf.Session()            
#c = GAN(sess,para_o)
#c.bulid_model()
#c.train('ionosphere')
#z = np.random.normal(size=[10,2])
#z = np.float32(z)
#ans = c.generator(z,reuse=True)
#sess.run(tf.Print(ans,[ans]))
