#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:16:00 2018

@author: tunan
"""
from keras.engine.topology import Layer
# for custom metrics
import keras.backend as K
import tensorflow as tf
import sys
sys.path.append('..')
from config import Config
opt  = Config()

class Attention(Layer):
    '''
    input = [v,q]
    v:[batch,15,v_dim]
    q:[batch,q_dim]

    return logit:[batch,15]
    h_dim,attention's dim = 1024
    1. 将q全连接到[batch,h_dim],将v全连接到[batch,15,h_dim]
    2.q,v点乘。加上dropout,再使用一个全连接到1。[batch,15,1]
    3.L2标准化，softmax
    '''
    # initialize the layer, and set an extra parameter axis. No need to include inputs parameter!
    def __init__(self,**kwargs):
        self.q_dim = 512
        self.v_dim = 512
        self.h_dim = 512
        self.drop_rate = 0.2
        self.feature_num = 30
        self.result = None
        super(Attention, self).__init__(**kwargs)

    # first use build function to define parameters, Creates the layer weights.
    # input_shape will automatic collect input shapes to build layer
    def build(self,input_shape):
        q_dim,v_dim,h_dim = self.q_dim,self.v_dim,self.h_dim
        self.q_proj = self.add_weight(name='q_kernel', 
                                      initializer='uniform',
                                      shape=(q_dim, h_dim),
                                      trainable=True)
        self.v_proj = self.add_weight(name = 'v_kernel',
                                      initializer='uniform',
                                      shape = (v_dim,h_dim),
                                      trainable = True)
        self.linear = self.add_weight(name = 'linear',
                                      initializer='uniform',
                                      shape = (h_dim,1),
                                      trainable =True)
        super(Attention, self).build(input_shape)

    # This is where the layer's logic lives. In this example, I just concat two tensors.
    def call(self, inputs, **kwargs):
        feature_num = self.feature_num
        [v,q] = inputs
        v_proj = K.tanh(K.dot(v,self.v_proj))
        q_proj = K.tanh(K.dot(q,self.q_proj))
        q_proj = K.expand_dims(q_proj,1)
        q_proj = tf.tile(q_proj,[1,feature_num,1])
        joint_repr = v_proj*q_proj
        joint_repr = K.dropout(joint_repr,self.drop_rate)
        
        logit = K.dot(joint_repr,self.linear)
        logit = K.reshape(logit,shape = [-1,feature_num])
        logit = K.softmax(K.l2_normalize(logit)) #[batch,K]
        logit = K.expand_dims(logit,-1)
        
        self.result = K.sum(logit*v,1)#v:[batch,K,4096]
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        print("output_shape",K.int_shape(self.result))
        return K.int_shape(self.result)


class Attention2(Layer):
    '''
    input = [v,q]
    v:[batch,15,v_dim]
    q:[batch,q_dim]

    return logit:[batch,15]
    h_dim,attention's dim = 1024
    1. 将q全连接到[batch,h_dim],将v全连接到[batch,15,h_dim]
    2.q,v点乘。加上dropout,再使用一个全连接到1。[batch,15,1]
    3.L2标准化，softmax
    '''
    # initialize the layer, and set an extra parameter axis. No need to include inputs parameter!
    def __init__(self,**kwargs):
        self.q_dim = 512
        self.v_dim = 512
        self.h_dim = 512
        self.drop_rate = 0.2
        self.feature_num = 30
        self.result = None
        super(Attention2, self).__init__(**kwargs)

    # first use build function to define parameters, Creates the layer weights.
    # input_shape will automatic collect input shapes to build layer
    def build(self,input_shape):
        q_dim,v_dim,h_dim = self.q_dim,self.v_dim,self.h_dim
        self.linear = self.add_weight(name = 'linear',
                                      initializer='uniform',
                                      shape = (h_dim,1),
                                      trainable =True)
        super(Attention2, self).build(input_shape)

    # This is where the layer's logic lives. In this example, I just concat two tensors.
    def call(self, inputs, **kwargs):
        '''
        v [batch,30,512]
        q [batch,512]
        '''
        feature_num = self.feature_num
        [v,q] = inputs
        q_proj = K.expand_dims(q,1)
        q_proj = tf.tile(q_proj,[1,feature_num,1])
        joint_repr = v*q_proj
        joint_repr = K.dropout(joint_repr,self.drop_rate)
        
        logit = K.dot(joint_repr,self.linear)
        logit = K.reshape(logit,shape = [-1,feature_num])
        logit = K.softmax(K.l2_normalize(logit)) #[batch,K]
        logit = K.expand_dims(logit,-1)
        
        self.result = K.sum(logit*v,1)#v:[batch,K,4096]
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        print("output_shape",K.int_shape(self.result))
        return K.int_shape(self.result)