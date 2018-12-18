#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 22:08:58 2018

@author: tunan
"""
import json,os
#所有路径的根目录在video下级
class Config(object):
    sequence_length = 19
    drop_out = 0.7
    embedding_size = 300
    class_num = 1000
    answer_num = 1000
    epochs = 30
    vgg_feature_dim = 4096      #2048,4096
    img_feature_dim = 2048
    pretrain_model = 'resnet152' #'vgg19','resnet152'
    
    loss = 'softmax'
    
    #data文件夹下所有文件的路径
    glove_path = 'data/glove.6B/glove.6B.300d.txt'
    train_text_path = 'data/train.txt'
    test_text_path = 'data/test.txt'
    train_video_path = 'data/train'      #训练视频的路径
    test_video_path = 'data/test'
    
    #模型mfl文件夹下的路径
    mflimage_frame = 5
    mfltrain_image_path = 'mfl/train_img'    #mfl模型训练图片的路径
    mfltest_image_path = 'mfl/test_img'    
    mflimg_to_index_path = 'mfl/img_to_index.json'    #视频名对应mfl特征矩阵中的序号
    mflvideo_feature_path = 'mfl/mfl_video_feature.h5'        #视频的特征，组成的矩阵、
    
    
    #utils下的路径
    wordans_to_index_path='utils/wordans_to_index.json'
    train_encode_path = "utils/train_encode.txt"
    val_encode_path = 'utils/val_encode.txt'
    embed_matrix_path = 'utils/embed_matrix.h5'
    
    
    def get_vocabularysize(self):
        wordans_to_index_path = os.path.join("/home/tunan/桌面/VQA",self.wordans_to_index_path)
        with open(wordans_to_index_path) as f:
            wordans_to_index = json.load(f)
        word_to_index = wordans_to_index['word_to_index']
        return len(word_to_index)
    def __init__(self):
        self.vocabulary_size = self.get_vocabularysize()
        

    
      
      
