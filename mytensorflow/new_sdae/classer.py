#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 21:33:38 2017

@author: zhouying
"""
import tensorflow as tf
#import numpy as np
import myutil
class mysdae(object):
    
    def __init__(self,input,hidden1,hidden2,transfer_function = tf.nn.softplus, 
                 optimizer = tf.train.AdamOptimizer(), 
                 keep_rate = 1, scale = 0.1):
        self.input_units = input.shape[0]
        self.output_units = self.input_units
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.transfer = transfer_function 
        self.scale = tf.placeholder(tf.float32) 
        self.training_scale = scale 
        self.keep_rate = tf.placeholder(tf.float32) 
        self.keep_prob = keep_rate
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.sess = tf.Session()
        
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['ew1'] = tf.Variable(myutil.xavier_init([self.input_units, self.hidden1]))
        all_weights['eb1'] = tf.Variable(tf.zeros([self.hidden1], dtype = tf.float32))
        all_weights['dw1'] = tf.Variable(tf.zeros([self.hidden1, self.input_units], dtype = tf.float32))
        all_weights['db1'] = tf.Variable(tf.zeros([self.input_units], dtype = tf.float32))
        all_weights['ew2'] = tf.Variable(myutil.xavier_init([self.hidden1, self.hidden2]))
        all_weights['eb2'] = tf.Variable(tf.zeros([self.hidden2], dtype = tf.float32))
        all_weights['dw2'] = tf.Variable(tf.zeros([self.hidden2, self.hidden1], dtype = tf.float32))
        all_weights['db2'] = tf.Variable(tf.zeros([self.hidden1], dtype = tf.float32))
        return all_weights
    
    def train1(self,train):
        z1 = self.transfer(tf.matmul(self.x,self.all_weights['ew1'])+self.all_weights['eb1'])
        o1 = self.transfer(tf.matmul(z1,self.all_weights['dw1'])+self.all_weights['db1'])
        self.loss1 = tf.reduce_mean((o1-self.x),2)
        self.sess.run((self.loss1, self.optimizer), feed_dict = {self.x: train,
                                                                            self.scale: self.training_scale,
                                                                            self.keep_rate: self.keep_prob
                                                                            })
    def train2(self,train):
        z2 = self.transfer(tf.matmul(self.x,self.all_weights['ew2'])+self.all_weights['eb2'])
        o2 = self.transfer(tf.matmul(z2,self.all_weights['dw2'])+self.all_weights['db2'])
        self.loss2 = tf.reduce_mean((o2-self.x),2)
        self.sess.run((self.loss2, self.optimizer), feed_dict = {self.x: train,
                                                                            self.scale: self.training_scale,
                                                                            self.keep_rate: self.keep_prob
                                                                            })
    def fine_tune(self,train):
        z1 = self.transfer(tf.matmul(self.x,self.all_weights['ew1'])+self.all_weights['eb1'])
        z2 = self.transfer(tf.matmul(z1,self.all_weights['ew2'])+self.all_weights['eb2'])
        o2 = self.transfer(tf.matmul(z2,self.all_weights['dw2'])+self.all_weights['db2'])
        o1 = self.transfer(tf.matmul(o2,self.all_weights['dw1'])+self.all_weights['db1'])
        self.loss3 = tf.reduce_mean((o1-self.x),2)
        self.sess.run((self.loss3, self.optimizer), feed_dict = {self.x: train,
                                                                            self.scale: self.training_scale,
                                                                            self.keep_rate: self.keep_prob
                                                                            })
    
    def re_error(self,data):
        z1 = self.transfer(tf.matmul(data,self.all_weights['ew1'])+self.all_weights['eb1'])
        z2 = self.transfer(tf.matmul(z1,self.all_weights['ew2'])+self.all_weights['eb2'])
        o2 = self.transfer(tf.matmul(z2,self.all_weights['dw2'])+self.all_weights['db2'])
        o1 = self.transfer(tf.matmul(o2,self.all_weights['dw1'])+self.all_weights['db1'])
        return data-o1
        