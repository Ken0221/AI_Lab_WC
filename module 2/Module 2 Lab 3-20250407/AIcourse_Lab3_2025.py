# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:19:52 2021

@author: Mint
"""

import math
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
    #return np.abs(np.array(result).reshape((-1, Nt*Nr*Ns)))

#比較不同的distribution: uniform, exponetial, 高斯
def get_noise_dataset(size, train_dataset, shape=(100), noise_type='uniform'):
    noise = np.zeros([len(train_dataset),Ns,Nt*Nr])
    for i in range(len(train_dataset)): 
        for j in range(Ns):
            for k in range(Nt*Nr):
                if noise_type == 'uniform':
                    noise[i,j,k] = random.uniform(-0.5, 0.5)
                elif noise_type == 'exponential':
                    noise[i,j,k] = np.random.exponential(0.2)
                elif noise_type == 'gaussian':
                    noise[i,j,k] = np.random.normal(0, 0.5)
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
    # your code
    c2 = tf.keras.layers.Conv1D(64, 5, activation=tf.nn.leaky_relu, kernel_constraint=w_clip, padding='same')(c1)
    c3 = tf.keras.layers.Conv1D(128, 5, activation=tf.nn.leaky_relu, kernel_constraint=w_clip, padding='same')(c2)
    c4 = tf.keras.layers.Conv1D(128, 5, activation=tf.nn.leaky_relu, kernel_constraint=w_clip, padding='same')(c3)
    f1= tf.keras.layers.Flatten()(c4)
    d1 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)(f1)
    d2 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(d1)
    d3 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(d2)
    d4 = tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu)(d3)
    d5 = tf.keras.layers.Dense(4, activation=tf.nn.leaky_relu)(d4)
    # your code
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(d5) 
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    return model
  
def get_generator(input_shape=(56,4), output_channel=4):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(4*1024*4, tf.nn.relu)(x)
    x = tf.keras.layers.Reshape((4, 1024*4))(x)
    x = tf.keras.layers.Conv1DTranspose(output_channel, 5, 2, padding='same', activation=tf.nn.relu)(x)
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

#loss 學生自己寫


def gen_train_step(noise):
    with tf.GradientTape() as tape:
        fake_data = model.generator(noise, training=True)
        fake = model.discriminator(fake_data, training=False)
        losses = -tf.reduce_mean(fake)
    gradients = tape.gradient(losses, model.generator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients, model.generator.trainable_variables))

    return losses

def dis_train_step(real_data, noise):
    with tf.GradientTape() as tape:
        fake_data = model.generator(noise, training=False)
        fake = model.discriminator(fake_data, training=True)
        real = model.discriminator(real_data, training=True)
        losses = -(tf.reduce_mean(real)-tf.reduce_mean(fake))
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
    train_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_1' for i in range(1,2) for p in ['db']]
    test_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_2' for i in range(1,2) for p in ['db']]
    
    sample_num = 200 #@param
    train_dataset = np.array([read_csi_samples(ds,sample_num) for ds in train_database])
    test_dataset = np.array([read_csi_samples(ds,sample_num) for ds in test_database])

    train_dataset = np.reshape(train_dataset,[dataset_num*sample_num,Ns,Nt*Nr])
    test_dataset = np.reshape(test_dataset,[dataset_num*sample_num,Ns,Nt*Nr])

    noise_types = ['uniform', 'exponential', 'gaussian']
    epoch = 20
    dis_loss = [np.zeros(epoch) for _ in range(len(noise_types))]
    gen_loss = [np.zeros(epoch) for _ in range(len(noise_types))]
    for i in range(len(noise_types)):
        print("noise_type:", noise_types[i])
        noise_dataset = get_noise_dataset(sample_num * len(train_database),train_dataset, noise_type=noise_types[i])
    # noise_dataset = get_noise_dataset(sample_num * len(train_database),train_dataset, )
        real_data = train_dataset
        noise = noise_dataset
    #%%    
        model = get_af_dcgan(clipping_value)
        model.generator.summary()
        model.discriminator.summary()
        gen_opt = tf.keras.optimizers.RMSprop()
        dis_opt = tf.keras.optimizers.RMSprop()
        
    #%%
        # epoch = 20
        # dis_loss = np.zeros(epoch)
        # gen_loss = np.zeros(epoch)
        for epochs in range(epoch):
            print("epochs:",epochs)
            dis_loss[i][epochs] = dis_train_step(real_data, noise)
            gen_loss[i][epochs] = gen_train_step(noise)
            print("dis_loss:",dis_loss[i][epochs])
            print("gen_loss:",gen_loss[i][epochs])
            
    #%%
        
        gen_data = model.generator.predict(noise)
        
    #%%    
        
        test_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_1' for i in range(1, 2) for p in ['db']]
        
        test_sample_num = 600 #@param
        test_dataset = np.array([read_csi_samples(ds,test_sample_num) for ds in test_database])
        dataset_num = 1
        test_dataset = np.reshape(test_dataset,[dataset_num*test_sample_num,Ns,Nt*Nr]) 
        
        test = np.vstack((test_dataset,gen_data))

        labels = np.zeros(len(test)) 
        for i in range(len(test_dataset)):
            labels[i] = 1
        testing_label = np.reshape(labels,(len(test),1))
    
    #%%
        Ans = model.discriminator.predict(test)
        Ans = np.around(Ans)
    #%%
        
        print('\nConfusion Matrix:')
        print(confusion_matrix(testing_label,Ans))

# draw loss curve
plt.figure(figsize=(5, 5))
plt.plot(dis_loss[0], label='uniform')
plt.plot(dis_loss[1], label='exponential')
plt.plot(dis_loss[2], label='gaussian')
plt.title('Discriminator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss') 
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(gen_loss[0], label='uniform')
plt.plot(gen_loss[1], label='exponential')
plt.plot(gen_loss[2], label='gaussian')
plt.title('Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# print(gen_loss[0])
# print(gen_loss[1])
# print(gen_loss[2])
# print(dis_loss[0])
# print(dis_loss[1])
# print(dis_loss[2])
    
    
    