#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:51:48 2018

@author: tunan
"""
import os,sys
import numpy as np
from tqdm import tqdm
import json,h5py

sys.path.append('..')
from config import Config


def get_result(model_num,prob_path):
    result = []
    for i in tqdm(range(model_num)):
        result_path = os.path.join(prob_path,'result_prob'+str(i+1)+'.h5')
        with h5py.File(result_path) as h:
            result_prob = np.array(h["result_prob"])
        result.append(np.expand_dims(result_prob,0))
    result = np.concatenate(result,axis = 0)
#    return np.mean(result,axis = 0)
    return result
   
def merge_two_prob():    
    #res attention
    prob_path = 'attmfl_res/prob_res'
    result1 = get_result(30,prob_path)
    
    #vgg attention
    prob_path = 'attmfl_c3d/prob_res'
    result2 = get_result(8,prob_path)
    
    result = np.concatenate([result1,result2],axis = 0)
    return np.mean(result,axis = 0)

def merge_average(prob1,prob2):
    prob1 = np.expand_dims(prob1,0)
    prob2 = np.expand_dims(prob2,0)
    result = np.concatenate([prob1,prob2],axis = 0)
    return np.mean(result,axis = 0)
    
if __name__ =="__main__":
    opt = Config()
    test_text_path = opt.test_text_path

    seq_length = opt.sequence_length
    
# =============================================================================
#     prob_path = 'attvcfl/mcfl_prob'
#     result_prob = get_result(24,prob_path)
#     submit_name = r'submits/attvc24.txt'
# =============================================================================
    
    #将attvcfl 和 attmfl的特征一起融合
    result_prob = merge_two_prob()
    submit_name = r'submits/c3d8res30.txt'
    
    with open(test_text_path,'r') as f:
        tests = f.readlines()
    result = []    
    with open(opt.wordans_to_index_path) as f:              
        metadata = json.load(f)
    ans_to_index = metadata['ans_to_index']
    index_to_answer = {ix:answer for answer,ix in ans_to_index.items()}
    
    index = 0
    for test in tqdm(tests):
        test = test.strip().split(',')
        line_result = []
        #提取图片特征
        name = test[0]
        question_indexs = [1,5,9,13,17]
        for question_index in question_indexs:
            #问题编码
            question = test[question_index]
            label = result_prob[index,:]
            index += 1
            line_result.append([question,index_to_answer[np.argmax(label)]])
#            print(question,":    ",answer_ix[np.argmax(label)])
#            input("next")
            
        predict_line = name +',' + ",".join([",".join(x) for x in line_result])
        result.append(predict_line)
    
    #写入文件
    result_text = '\n'.join(result)
    with open(submit_name,'w') as f:
        f.write(result_text)
