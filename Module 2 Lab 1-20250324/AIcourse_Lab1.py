# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#下載mnist資料集檔案 資料集檔案位置:C:\Users\.keras\datasets\mnist.npz
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
np.random.seed(10)

#%%
if __name__ == '__main__':
    #load data
    (x_train_image,y_train_label),\
    (x_test_image,y_test_label) = mnist.load_data()
    
    #reshape
    x_train_imre = x_train_image.reshape(x_train_image.shape[0], x_train_image.shape[1]*x_train_image.shape[2]).astype('float32')
    x_test_imre = x_test_image.reshape(x_test_image.shape[0], x_test_image.shape[1]*x_test_image.shape[2]).astype('float32')
    
    #normalize
    x_train_normalize = x_train_imre/255
    x_test_normalize = x_test_imre/255
    
    #one hot
    y_TrainOneHot = np_utils.to_categorical(y_train_label)
    y_TestOneHot = np_utils.to_categorical(y_test_label)
    
    #%% 加分 畫train data的圖
    # plt.imshow

    # your code
    plt.figure(figsize=(10,4))
    for i in range(0,10):
        for j in range(x_train_image.shape[0]):
            if y_train_label[j] == i:
                plt.subplot(2,5,i+1)
                plt.imshow(x_train_image[j], cmap='gray')
                plt.title('label=' + str(y_train_label[j]))
                plt.axis('off')
                break
    
    plt.suptitle("MNIST Training Data Examples", fontsize=16)
    plt.show()
    
    #%%
    #build model
    inputs = keras.Input(shape = 784) 
    # your code
    hidden1 = tf.keras.layers.Dense(512, activation='relu')(inputs)
    hidden2 = tf.keras.layers.Dense(256, activation='relu')(hidden1)
    hidden3 = tf.keras.layers.Dense(128, activation='relu')(hidden2)
    hidden4 = tf.keras.layers.Dense(64, activation='relu')(hidden3)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(hidden4)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    #start training
    train_history = model.fit(x=x_train_normalize, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=2)
    
    def show_train_history(train_history, train, validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('train history')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    show_train_history(train_history, 'accuracy', 'val_accuracy')
    show_train_history(train_history, 'loss', 'val_loss')
    
    scores = model.evaluate(x_test_normalize, y_TestOneHot)
    print('test loss, test accuracy=', scores)
    
    
    #%% confusion matrix
    
    # your code
    # 預測測試資料集
    y_pred = model.predict(x_test_normalize)
    y_pred_classes = np.argmax(y_pred, axis=1)  # 預測類別
    y_true = np.argmax(y_TestOneHot, axis=1)    # 真實類別

    # 計算混淆矩陣
    conf_matrix = confusion_matrix(y_true, y_pred_classes)

    # 視覺化混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
        