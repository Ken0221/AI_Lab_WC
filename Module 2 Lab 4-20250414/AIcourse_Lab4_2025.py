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
from sklearn.preprocessing import OneHotEncoder, normalize
import matplotlib.pyplot as plt
#%%
def read_csi_samples(filename,sample_num):
    data = sio.loadmat(filename)
    result = []
    for d in data['original_csi']:
        result.append(d[0]['csi'][0][0])
        result = result[0:sample_num]
    return np.abs(np.array(result).reshape((-1, Nt*Nr, Ns)).transpose((0, 2, 1)))

def get_noise_dataset(size, shape=(100)):
    noise = np.zeros(shape)
    for i in range(100):        
        noise[i] = random.random()
    noise = noise.transpose()
    return noise


class ClipConstrain(tf.keras.constraints.Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value
    
    def __call__(self, weight):
        return tf.clip_by_value(weight, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}

def get_discriminator(clipping_value):
    w_clip = ClipConstrain(clipping_value)
    inputs = keras.Input(shape = (56,4))
    c1 = tf.keras.layers.Conv1D(32, 5, activation=tf.nn.leaky_relu, kernel_constraint=w_clip, padding='same')(inputs)
    c2 = tf.keras.layers.Conv1D(64, 5, activation=tf.nn.leaky_relu, kernel_constraint=w_clip, padding='same')(c1)
    c3 = tf.keras.layers.Conv1D(128, 5, activation=tf.nn.leaky_relu, kernel_constraint=w_clip, padding='same')(c2)
    c4 = tf.keras.layers.Conv1D(128, 5, activation=tf.nn.leaky_relu, kernel_constraint=w_clip, padding='same')(c3)
    f1= tf.keras.layers.Flatten()(c4)
    d1 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)(f1)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(d1) 
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    return model

def get_generator(input_shape=(100), output_channel=4):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Dense(4*1024*4, tf.nn.relu)(inputs)
    x = tf.keras.layers.Reshape((4, 1024*4))(x)
    x = tf.keras.layers.Conv1DTranspose(128, 5, 2, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv1DTranspose(64, 5, 2, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv1DTranspose(32, 5, 2, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv1DTranspose(output_channel, 5, 2, padding='same', activation=tf.nn.tanh)(x)
    f1= tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(224, tf.nn.relu)(f1)
    outputs = tf.keras.layers.Reshape((56,4))(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
    
class AFDCGAN():
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

def get_af_dcgan(clipping_value):
    generator = get_generator()
    discriminator = get_discriminator(clipping_value)
    return AFDCGAN(generator, discriminator)

def gen_train_step(noise):
    with tf.GradientTape() as tape:
        fake_data = model.generator(noise, training=True)
        fake = model.discriminator(fake_data, training=False)
        losses = - tf.reduce_mean(fake)
    gradients = tape.gradient(losses, model.generator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients, model.generator.trainable_variables))

    return losses

def dis_train_step(real_data, noise):
    with tf.GradientTape() as tape:
        fake_data = model.generator(noise, training=False)
        fake = model.discriminator(fake_data, training=True)
        real = model.discriminator(real_data, training=True)
        losses = - tf.reduce_mean(real) + tf.reduce_mean(fake)
    gradients = tape.gradient(losses, model.discriminator.trainable_variables)
    dis_opt.apply_gradients(zip(gradients, model.discriminator.trainable_variables))

    return losses
#%%

if __name__ == '__main__':
        
    
    ROOT_DIR = 'C:/Users/kenla/Desktop/113-2/AI_Lab_WC'
    
    Nap, Nt, Nr, Ns, M = 2, 2, 2, 56, 6
    dataset_num = 1
    
    clipping_value = 1e-2 #@param
    
    
    """##Dataset"""
  
    train_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_1' for i in range(1,2) for p in ['a']]
    test_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_2' for i in range(1,2) for p in ['a']]
    
    sample_num = 200 #@param
    train_dataset = np.array([read_csi_samples(ds,sample_num) for ds in train_database])
    test_dataset = np.array([read_csi_samples(ds,sample_num) for ds in test_database])

    train_dataset = np.reshape(train_dataset,[dataset_num*sample_num,Ns,Nt*Nr])
    test_dataset = np.reshape(test_dataset,[dataset_num*sample_num,Ns,Nt*Nr])
    
    for i in range(len(train_dataset)):
        train_dataset[i,:,:] = normalize(train_dataset[i,:,:], axis= 1)
    
    for i in range(len(test_dataset)):
        test_dataset[i,:,:] = normalize(test_dataset[i,:,:], axis= 1)
    
    noise_dataset = np.zeros([200,100])
    for i in range(200):
        noise_dataset[i] = get_noise_dataset(sample_num * len(train_database))
    real_data = train_dataset
    noise = noise_dataset
#%%    
    model = get_af_dcgan(clipping_value)
    model.generator.summary()
    model.discriminator.summary()
    
#%%

    gen_opt = tf.keras.optimizers.RMSprop()
    dis_opt = tf.keras.optimizers.RMSprop()
    
    
    for epochs in range(20):
        print("epochs:",epochs+1)
        for min_epochs in range(5):
            dis_loss = dis_train_step(real_data, noise)
        gen_loss = gen_train_step(noise)
        print("dis_loss:",dis_loss)
        print("gen_loss:",gen_loss)
        print("\n")
        
#%%
    
    gen_data = model.generator.predict(noise)
    
#%%    
    new_data = np.vstack((train_dataset,gen_data))
    data_len = len(new_data)
    
#%%    
    
    train_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_1' for i in range(1, 3) for p in ['a']]
    train_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_1' for i in range(2, 3) for p in ['a']]
    test_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_2' for i in range(1, 3) for p in ['a']]
    
    train_sample_num = 400 #@param
    train_dataset = np.array([read_csi_samples(ds,train_sample_num) for ds in train_database])    
    dataset_num = 1*1
    train_dataset = np.reshape(train_dataset,[dataset_num*train_sample_num,Ns,Nt*Nr])
    
    test_sample_num = 600 #@param
    test_dataset = np.array([read_csi_samples(ds,test_sample_num) for ds in test_database])
    dataset_num = 2*1
    test_dataset = np.reshape(test_dataset,[dataset_num*test_sample_num,Ns,Nt*Nr])

    for i in range(len(train_dataset)):
        train_dataset[i,:,:] = normalize(train_dataset[i,:,:], axis= 1)    
    for i in range(len(test_dataset)):
        test_dataset[i,:,:] = normalize(test_dataset[i,:,:], axis= 1)    
    
    train_dataset = np.vstack((new_data,train_dataset))
      
    idx = [i for i in range(len(train_dataset))]
    random.shuffle(idx)
     
    labels = np.zeros(len(train_dataset))
    for i in range(train_sample_num):
        labels[i] =  1
    training_label = np.reshape(labels,(len(train_dataset),1))
     
    dataset_num = 2*1
    labels = np.zeros(dataset_num*test_sample_num) 
    for i in range(test_sample_num):
        labels[i] = 1
    testing_label = np.reshape(labels,(dataset_num*test_sample_num,1))
#%%    
    onehotencoder = OneHotEncoder()
    training_label = onehotencoder.fit_transform(training_label).toarray()
    testing_label = onehotencoder.fit_transform(testing_label).toarray()
    training_label = training_label[idx]
    train_dataset = train_dataset[idx]

#%%    
    inputs = keras.Input(shape = (56,4))
    #-----------------------------------------------------
    # your code
    #-----------------------------------------------------
    c1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='leaky_relu', padding='same', input_shape=(56,4))(inputs)
    c2 = tf.keras.layers.MaxPooling1D(pool_size=2)(c1)
    # c2 = tf.keras.layers.BatchNormalization()(c2)
    # c2 = tf.keras.layers.Dropout(0.2)(c2)
    c3 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='leaky_relu', padding='same')(c2)
    c4 = tf.keras.layers.MaxPooling1D(pool_size=2)(c3)
    # c4 = tf.keras.layers.BatchNormalization()(c4)
    # c4 = tf.keras.layers.Dropout(0.2)(c4)
    c5 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='leaky_relu', padding='same')(c4)
    c6 = tf.keras.layers.MaxPooling1D(pool_size=2)(c5)
    # c6 = tf.keras.layers.BatchNormalization()(c6)
    # c6 = tf.keras.layers.Dropout(0.2)(c6)
    c7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='leaky_relu', padding='same')(c6)
    c8 = tf.keras.layers.MaxPooling1D(pool_size=2)(c7)
    # c8 = tf.keras.layers.BatchNormalization()(c8)
    # c8 = tf.keras.layers.Dropout(0.2)(c8)
    f1 = tf.keras.layers.Flatten()(c8)
    d1 = tf.keras.layers.Dense(256, activation='leaky_relu')(f1)
    d2 = tf.keras.layers.Dense(64, activation='leaky_relu')(d1)
    d3 = tf.keras.layers.Dense(6, activation='softmax')(d2)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(d3)
    
    model2 = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model2.compile(loss="categorical_crossentropy", optimizer = 'Adam', metrics=['accuracy'])    
    history = model2.fit(train_dataset, training_label, epochs=15, validation_split= 0.1)
    model2.summary()
    
#%%   
    model2.evaluate(test_dataset, testing_label) 
    y_pred = onehotencoder.inverse_transform(model2.predict(test_dataset))  
    y_true = onehotencoder.inverse_transform(testing_label)
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