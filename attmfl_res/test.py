#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:59:13 2018

@author: tunan
"""


import os,sys
import keras.models 
import numpy as np
from tqdm import tqdm
import cv2,json,datetime,h5py
from matplotlib import pyplot as plt
sys.path.append('..')
from config import Config
from utils.pretreatment import pretreat_embedAndText
from pretrain_model.feature_extract import get_pretrain_model
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

def extract_video_frame(name,image_num):
    vc = cv2.VideoCapture(name) #读入视频文件
    #将视频读取到列表中
    imgs = []
    rval, frame = vc.read()
    if not rval:
        print('视频文件不存在')
        return 
    
    while rval:
        imgs.append(frame)
        rval,frame = vc.read()
    vc.release()
    
    print(f'帧数为：{len(imgs)}')
    step = len(imgs)/float(image_num)
    fms = np.zeros(shape = [30,224,224,3])
    #now 只能取到image_num-1,故imgs最多取到len(imgs)-1，不用担心越界
    for now in range(image_num):
        frame = imgs[int(now*step)]
        fms[now,...] = cv2.resize(frame,(224, 224),fx=0, fy=0,interpolation=cv2.INTER_AREA)
    return fms

def main(**kwargs):
    tsvideo = 'test/1.mp4'
    model_path = 'attmfl_res/models/1'
    if 'video' in kwargs:
        tsvideo = kwargs['video']
    if 'model_path' in kwargs:
        model_path = kwargs['model_path']
    
    #提取30个视频帧
    fms = extract_video_frame(tsvideo,30)
    plt.figure('show mp4',figsize = (8,8))    #figsize是每个子窗口的大小，也就是每个图片显示的分辨率
    for i in range(30):
        plt.subplot(6,5,i+1)    #前两个参数将图片划分成4*4的窗口，i+1表示在第几个窗口显示
        plt.title(f'image {i+1}')
        plt.axis('off')
        plt.imshow(fms[i,...].astype(np.uint8))
    plt.show()
    
    #提取视频帧的特征。
    print('start extract feature from images')
    print('---------------------------------')
    model,lambda_func = get_pretrain_model('resnet152')       
    fms = lambda_func(fms)
    feature = model.predict(fms)
    print('finished!')
    
    questions = ['what is in the video',
                 'where is the person in the video',
                 'what color clothes does the person wear in the video']

    encoder = pretreat_embedAndText()
    with open('utils/wordans_to_index.json') as f:              
        wordans_to_index = json.load(f)
    word_to_index = wordans_to_index['word_to_index']
    answer_to_index = wordans_to_index['ans_to_index']
    idx_to_asw = {answer_to_index[key]:key for key in answer_to_index}
    word = encoder.convert_question(questions[1],word_to_index,seq_lenth = 19)
    #vqa模型
    vqa_model = keras.models.load_model(get_model_path(model_path),
                                         custom_objects={'my_accuracy': my_accuracy,
                                                         'Attention':Attention()})
    
    x = 'y'
    i = 0
    while(True):
        x = input('please input question!\ntype n exist,\ntype r use predefine question\ninput:')
        if x == 'r':
            x = questions[i]
            i = (i+1)%3
        if x == 'n':
            break
        
        word = encoder.convert_question(x,word_to_index,seq_lenth = 19)
        prob = vqa_model.predict([np.expand_dims(feature,0),word.reshape(1,-1)])
        asw = idx_to_asw[np.argmax(prob)]    
        print('----------------------\n')
        print(f'question:\t{x}\nanswer:\t {asw}')
        print('------------------------')
        
if __name__ =="__main__":
    os.chdir('..')
    import fire
    fire.Fire(main)
