#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:52:21 2017

@author: zhouying
"""

def mlp(x_train,y_train,x_test):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
#    from keras.optimizers import RMSprop
    model = Sequential() 
    model.add(Dense(10, input_shape=(8,)))
    model.add(Activation('sigmoid')) # ReLU
#    model.add(Dropout(0.2)) # Dropout
#    model.add(Dense(10)) # 全连接层
#    model.add(Activation('sigmoid')) # ReLU
#    model.add(Dropout(0.2)) # Dropout
    model.add(Dense(1)) # 分类层
    model.add(Activation('softmax')) # Softmax
    model.summary() # 打印模型
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=10)
    classes = model.predict(x_test, batch_size=1)
    return classes 