#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:16:46 2018

@author: tunan
"""

import h5py,os,sys
import numpy as np
from keras.utils.np_utils import to_categorical
sys.path.append('..')
from config import Config

opt = Config() 

#输入视频的名字，返回他的特征 [2,15*4096],第一个是vgg,第二个是c3d       
def get_feature(name,index):
    feature_path = os.path.join('features/resnet152',str(index),'train')
    path = os.path.join(feature_path,name)
    with h5py.File(path + '.h5') as h:
        feature = np.array(h['img_feature'])
    return feature
        
class my_dataset(object):        
    def generate_data(self,train_encode_path,batch_size = 64):
        '''
        train_encode_path: 编码的txt文件，记录(img_name,encode_question,ans1,2,3)
        '''
        index = 1
        start = 0
        with open(train_encode_path) as f:
            train_data = f.readlines()
            
        while True:
            if start+batch_size<len(train_data):
                end = start+batch_size 
            else:
                end = len(train_data)
                
            questions= [] 
            answers = []
            images = []
            for line in range(start,end):
                sample = train_data[line].strip().split(',')
                name = sample[0]
                questions.append(np.array(sample[1:20],dtype = np.int32).reshape([1,-1]))
                answers.append(np.array(sample[20:],dtype = np.int32).reshape([1,-1]))
                images.append(get_feature(name,index).reshape(1,-1,opt.img_feature_dim))
            questions = np.concatenate(questions,axis = 0)
            answers = np.concatenate(answers,axis = 0)
            images = np.concatenate(images,axis=0 )
                
            start = end
            #一个epoch结束,更换下一个feature源,index代表当前使用哪个feature
            if start == len(train_data):
                start = 0
                index = index+1 if index <5 else 1
                
            answers_onehot = to_categorical(answers,1000)
            answers_onehot = np.max(answers_onehot,axis = 1)
            yield [images,questions],answers_onehot
            
    def generate_val(self,train_encode_path,batch_size = 64):
        '''
        train_encode_path: 编码的txt文件，记录(img_name,encode_question,ans1,2,3)
        '''
        start = 0
        with open(train_encode_path) as f:
            train_data = f.readlines()
            
        while True:
            if start+batch_size<len(train_data):
                end = start+batch_size 
            else:
                end = len(train_data)
                
            questions= [] 
            answers = []
            images = []
            for line in range(start,end):
                sample = train_data[line].strip().split(',')
                name = sample[0]
                questions.append(np.array(sample[1:20],dtype = np.int32).reshape([1,-1]))
                answers.append(np.array(sample[20:],dtype = np.int32).reshape([1,-1]))
                images.append(get_feature(name,1).reshape(1,-1,opt.img_feature_dim))
            questions = np.concatenate(questions,axis = 0)
            answers = np.concatenate(answers,axis = 0)
            images = np.concatenate(images,axis=0 )
                
            start = end
            #一个epoch结束,更换下一个feature源,index代表当前使用哪个feature
            if start == len(train_data):
                start = 0
                
            answers_onehot = to_categorical(answers,1000)
            answers_onehot = np.max(answers_onehot,axis = 1)
            yield [images,questions],answers_onehot
            
if __name__ == '__main__':            
    os.chdir('..')
    opt = Config()
    gen = my_dataset().generate_data(opt.train_encode_path)
    x = next(gen)
    print(x[0][0].shape,x[0][1].shape)
