#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:31:59 2018

@author: tunan
"""

import os,sys
import keras.models 
import numpy as np
from tqdm import tqdm
import cv2,json,datetime,h5py

sys.path.append('..')
from config import Config
from utils.pretreatment import pretreat_embedAndText
from attention import Attention
# for custom metrics
import keras.backend as K
#定义自己的评价函数
def my_accuracy(y_true,y_pred):
    newy_pred = K.one_hot(K.argmax(y_pred,axis = -1),y_pred.get_shape()[-1])
    compare = K.cast(K.minimum(newy_pred,y_true),K.floatx())    #只有同时是1，才会取1
    max_vector = K.max(compare ,axis = -1)
    return K.mean(max_vector)

def get_model_path(path):
    path_names = os.listdir(path)
    epochs = [int(name[6:9]) for name in path_names]
    path_name = path_names[np.argmax(epochs)]
    return os.path.join(path,path_name)

if __name__ =="__main__":
    os.chdir('..')
    opt = Config()
    for i in range(30):
        model_num = i+1
        
        #模型参数配置,model_path,prob_path,testfeature_path，
        model_path = os.path.join('attmfl_res/models',str(model_num))
        
        prob_path = 'attmfl_res/prob_res'
        testfeature_path = 'features/resnet152/2/test'
        img_num = 30

        #固定参数，无需改变
        result_prob_path = os.path.join(prob_path,'result_prob'+str(model_num)+'.h5')
        test_text_path = opt.test_text_path
        seq_length = opt.sequence_length
        if not os.path.exists(prob_path):
            os.makedirs(prob_path)
        #vqa模型
        vqa_model = keras.models.load_model(get_model_path(model_path),
                                             custom_objects={'my_accuracy': my_accuracy,'Attention':Attention()})
        
        with open(test_text_path,'r') as f:
            tests = f.readlines()
        result = []  #每个答案对应的概率都放入，5×len(tests)
        
        with open(opt.wordans_to_index_path) as f:              
            metadata = json.load(f)
        word_to_index = metadata['word_to_index']
        ans_to_index = metadata['ans_to_index']
        index_to_answer = {ix:answer for answer,ix in ans_to_index.items()}
        pretreator = pretreat_embedAndText()    
        
        for test in tqdm(tests):
            test = test.strip().split(',')
            #提取图片特征
            name = test[0]
            path = os.path.join(testfeature_path,name)
            if not os.path.exists(path+'.h5'):
                raise('not exist',path)
            with h5py.File(path+'.h5') as h:
                features = np.array(h['img_feature']).reshape([1,img_num,-1])
                
            question_indexs = [1,5,9,13,17]
            for question_index in question_indexs:
                #问题编码
                question = test[question_index]
                question_encode = pretreator.convert_question(
                        question,word_to_index,seq_lenth = seq_length).reshape([1,-1])
        
                label = vqa_model.predict([features,question_encode])
                result.append(label)
    
        #写入文件
        result = np.concatenate(result,axis = 0)
        with h5py.File(result_prob_path) as h:
            h.create_dataset("result_prob",data = result)
            
        print(get_model_path(model_path))
