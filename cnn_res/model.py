#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:18:23 2018

@author: tunan
"""

from keras.layers import Dense,Dropout,LSTM,Embedding,Input,multiply,Masking,BatchNormalization,Conv1D,MaxPool1D,add
from keras.models import Sequential,Model
import numpy as np
import sys,os,h5py
sys.path.append('..')
from config import Config
from attention import Attention2
# for custom metrics
import keras.backend as K

'''
Embedding层输入二维，输出三维.
形如（samples，sequence_length）的2D张量 ,输出shape 
形如(samples, sequence_length, output_dim)的3D张量
'''
#定义自己的评价函数
def my_accuracy(y_true,y_pred):
    newy_pred = K.one_hot(K.argmax(y_pred,axis = -1),y_pred.get_shape()[-1])
    compare = K.cast(K.minimum(newy_pred,y_true),K.floatx())    #只有同时是1，才会取1
    max_vector = K.max(compare ,axis = -1)
    return K.mean(max_vector)

def get_embed_matrix():
    opt = Config()
    embed_matrix_path = opt.embed_matrix_path
    if not os.path.exists(embed_matrix_path):
        raise("don't exists embed matrix,please run build_embed_matrix!")
    with h5py.File(embed_matrix_path) as h:
        embed_matrix = np.array(h['embed_matrix'])
    return embed_matrix

class cnn_res(object):
    def __init__(self,opt):
        self.opt = opt
        
    def wordtoVec_model(self,embedding_matrix,word_num,embedding_size,seq_length,drop_rate):
        model = Sequential()
        model.add(Masking(mask_value= -1,input_shape=(seq_length,)))
        model.add(Embedding(input_dim = word_num ,output_dim = embedding_size,
                            weights = [embedding_matrix],
                            input_length = seq_length,
                            trainable = False))
        model.add(LSTM(units=512,input_shape = (seq_length,embedding_size),return_sequences=True))
        model.add(Dropout(drop_rate))
        model.add(LSTM(units = 512,return_sequences=False))       #这一层为什么不用input_dim,会从上一层的值进行判断？
        return model
    
    def resblock(self):
        inputs = Input(shape = (30,2048))
        d = Dense(512,activation='tanh')(inputs)
        x = Conv1D(512,kernel_size=3,strides=1,activation='tanh',padding='same')(d)
        x = MaxPool1D(pool_size=3,strides=1,padding = 'same')(x)
        x = Conv1D(512,kernel_size=3,strides=1,activation='tanh',padding='same')(x)
        y = add([d,x])
        y = Dropout(0.7)(y)
        model = Model(inputs = inputs,outputs = y)
        return model
        
    def vqa_model(self,embedding_matrix,word_num,embedding_size,img_feature_dim,
                  seq_length,drop_rate,class_num,num_picture,loss):        
        vgg = Input(shape = (num_picture,img_feature_dim,))
        img_model = self.resblock()
        img = img_model(vgg)    #[batch,30,512]

        Embedding_model = self.wordtoVec_model(embedding_matrix,word_num,embedding_size,seq_length,drop_rate)
        q_input = Input(shape = (seq_length,))
        question = Embedding_model(q_input) #[batch,512]
    
        att_v = Attention2()([img,question]) #[batch,4096]
        
#        v_proj = Dense(512,activation = 'tanh')(att_v)
#        q_proj = Dense(512,activation = 'tanh')(question)
        
        print("starting merging...")
        mul = multiply([att_v,question])
        x = Dense(class_num,activation = 'tanh')(mul)
        x = Dropout(drop_rate)(x)
#        x = BatchNormalization()(x)
        x = Dense(class_num,activation = loss)(x)
        model = Model(inputs = [vgg,q_input],outputs = [x])
        model.compile(optimizer = 'rmsprop',loss='categorical_crossentropy',
                      metrics = ['accuracy',my_accuracy])
        return model
    
    def build_model(self):
        opt = self.opt      #load configuration parameter
        word_num = opt.vocabulary_size
        embedding_size = opt.embedding_size
        seq_length = opt.sequence_length
        drop_rate = opt.drop_out
        num_picture = 30
        class_num = opt.class_num
        loss = opt.loss
        img_feature_dim = opt.img_feature_dim
        
        embed_matrix = get_embed_matrix()
        model = self.vqa_model(embedding_matrix= embed_matrix,
                word_num=word_num,embedding_size=embedding_size,seq_length=seq_length,
                img_feature_dim = img_feature_dim,drop_rate=drop_rate,
                class_num = class_num,num_picture=num_picture,loss = loss)
        return model
        
if __name__ == '__main__':
    opt = Config()
    os.chdir("..")
    model = cnn_res(opt).build_model()
    print(model.summary())
