# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 08:59:07 2021

@author: Mint
"""

import random
import datetime
from functools import reduce
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
#%%
def read_csi_samples(filename,sample_num):
    data = sio.loadmat(filename)
    result = []
    for d in data['original_csi']:
        result.append(d[0]['csi'][0][0])
        result = result[0:sample_num]
    return np.abs(np.array(result).reshape((-1, Nt*Nr, Ns)).transpose((0, 2, 1)))
#%%

if __name__ == '__main__':
        
    
    ROOT_DIR = '/content'
    
    Nap, Nt, Nr, Ns, M = 2, 2, 2, 56, 6
    dataset_num = 6*4
    
    """##Dataset"""
    
    train_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_1' for i in range(1, 7) for p in ['db', 'a', 'c', 'j']]
    test_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_2' for i in range(1, 7) for p in ['db', 'a', 'c', 'j']]
  
    
    sample_num = 600 #@param
    train_dataset = np.array([read_csi_samples(ds,sample_num) for ds in train_database])
    test_dataset = np.array([read_csi_samples(ds,sample_num) for ds in test_database])


    train_dataset = np.reshape(train_dataset,[dataset_num*sample_num,Ns,Nt*Nr])
    test_dataset = np.reshape(test_dataset,[dataset_num*sample_num,Ns,Nt*Nr])
    
    labels = np.zeros(dataset_num*sample_num)
    
    for i in range(dataset_num):
        for j in range(sample_num):
            temp = i % 6
            labels[(i-1)*sample_num+j] =  i % 6
    labels = np.reshape(labels,(dataset_num*sample_num,1))
#%%
    
    onehotencoder = OneHotEncoder()
    training_label = onehotencoder.fit_transform(labels).toarray()
    test_label = training_label
    
#%%
    #加分題 打亂資料順序
    
    #your code

    
#%%

    #加分: Maxpooling/ batch normalize/ dropout
    #先完成基礎的CNN+DNN，再開始進行加分題
    
    inputs = keras.Input(shape = (56,4))

    #your code

#%%    
    model.evaluate(test_dataset, test_label, batch_size = 256) 
    y_pred = onehotencoder.inverse_transform(model.predict(test_dataset))  
    y_true = onehotencoder.inverse_transform(test_label)    
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_true,y_pred))

#%%
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()