#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:54:29 2018

@author: tunan
"""

import os,sys
import numpy as np
np.random.seed(2018)
import keras
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam,rmsprop

sys.path.append('..')
from config import Config
from my_dataset import my_dataset
from model import my_accuracy,attmfl
from attention import Attention

def get_model_path(path):
    path_names = os.listdir(path)
    epochs = [int(name[6:9]) for name in path_names]
    path_name = path_names[np.argmax(epochs)]
    return os.path.join(path,path_name)

def del_model_path(path):
    path_names = os.listdir(path)
    epochs = [int(name[6:9]) for name in path_names]
    last_path = path_names[np.argmax(epochs)]
    #delete all models except last one
    for p in path_names:
        if p is not last_path:
            os.remove(os.path.join(path,p))
            
def main(**kwargs):
    os.chdir('..')
    opt = Config()
    for key,value in kwargs.items():
        setattr(opt,key,value)
        
    model_builder = attmfl(opt)
    model = model_builder.build_model()
    
    #每次换模型都要改变的参数
    dataset = my_dataset()
    BATCH_SIZE = 128
    
    if 'model_path' in kwargs:
        model_path = kwargs['model_path']
    else :
        model_path = 'attmfl_res/models'
    train_encode_path = opt.train_encode_path
    val_encode_path = opt.val_encode_path
        
    #得到训练,测试样本数。以及每次需要训练的步数
    with open(opt.train_encode_path,'r') as f:
        text = f.readlines()
    Train_num = len(text)
    print('%d train samples'%(Train_num))
    with open(opt.val_encode_path,'r') as f:
        text = f.readlines()
    Val_num = len(text)
    print('%d validation samples'%(Val_num))
    train_step = np.ceil(Train_num/BATCH_SIZE)
    val_step = np.ceil(Val_num/BATCH_SIZE)
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_gen = dataset.generate_data(train_encode_path,batch_size = BATCH_SIZE)
    val_gen = dataset.generate_val(val_encode_path,batch_size = BATCH_SIZE)
    checkpoint_path = os.path.join(model_path,
                                   'epoch_{epoch:03d}val_acc{val_acc:.3f}val_myacc{val_my_accuracy:.4f}.h5')
    
    checkpoint = ModelCheckpoint(
            checkpoint_path,monitor='val_my_accuracy',
            save_best_only=True,save_weights_only=False,verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_accuracy', factor=0.3, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_my_accuracy', min_delta=0, patience=10, verbose=1)
    if True:
        model.compile(optimizer=rmsprop(lr=1e-3), loss="categorical_crossentropy",
                      metrics = ['accuracy',my_accuracy])
        model.fit_generator(train_gen,steps_per_epoch = train_step,
                            epochs=30,
                            validation_data=val_gen,
                            validation_steps=val_step,
                            shuffle=True,callbacks=[checkpoint])
    
    #load best model in pretrain
    if True:
        model = keras.models.load_model(get_model_path(model_path),
                                custom_objects={'my_accuracy': my_accuracy,'Attention':Attention()})
        model.layers[2].layers[1].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                      metrics = ['accuracy',my_accuracy]) # recompile to apply the change
        print('Unfreeze Embedding layers.')
        
        model.fit_generator(train_gen,steps_per_epoch = train_step,epochs=60,
                            initial_epoch = 30,
                            validation_data=val_gen,
                            validation_steps=val_step,
                            shuffle=True,callbacks=[checkpoint,reduce_lr,early_stopping])
        
    #删除其他无用模型
    del_model_path(model_path)


if __name__ == '__main__': 
    for i in range(30):        
        main(attmflmodel_path = ('models/attmfl'+str(i+1)))
#    import fire
#    fire.Fire(main)
