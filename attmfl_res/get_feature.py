#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:13:53 2018

@author: tunan
"""

import os,sys
sys.path.append('..')
from pretrain_model.feature_extract import imgFeatureExtract

def get_feature(**kwargs):
    #input image_path and feature path
    if 'image_path' in kwargs:
        image_path = kwargs['image_path']
    else:
        image_path = 'images/image1'
        
    if 'feature_path' in kwargs:
        feature_path = kwargs['feature_path']
    else:
        feature_path = 'features/resnet152/1'
        
    feature_extractor = imgFeatureExtract()
    #train feature
    model_name = 'resnet152'
    image_num = 30
    feature_dim = 2048
    trainimg_path = os.path.join(image_path,'train_images')
    trainfeature_path = os.path.join(feature_path,'train')
    feature_extractor.train_feature_extract(trainfeature_path,trainimg_path,model_name,image_num,feature_dim)
    
    #test feature
    testimage_path = os.path.join(image_path,'test_images')
    testfeature_path = os.path.join(feature_path,'test')
    feature_extractor.test_feature_extract(testfeature_path,testimage_path,model_name,image_num,feature_dim)

    
#注意inception_resnet_v2的特征数是1536
if __name__ == '__main__':
    os.chdir('..')
    import fire
    fire.Fire(get_feature)