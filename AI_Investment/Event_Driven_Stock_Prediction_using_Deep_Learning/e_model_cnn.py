#!/usr/bin/python
import random
import numpy as np
# import operator
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Flatten
# from keras.layers import Dropout
# from keras.layers.convolutional import Convolution2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.layers.convolutional import Convolution1D
# from keras.layers.convolutional import MaxPooling1D
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence

##### !!! NOTICE: input1為初版沒有考慮company embedding !!! #####

### 同matrix不同label，還是會出現？？？ ### => 合併漲跌

### 選擇適當的quantile threshold ###

from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


import pandas as pd
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers, models
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import time
# import seaborn as sns

from wakepy import set_keepawake, unset_keepawake
set_keepawake(keep_screen_awake=True)

def value2int(y, clusters=2):
    label = np.copy(y)
    label[y < np.percentile(y, 100 / clusters)] = 0
    for i in range(1, clusters):
        label[y > np.percentile(y, 100 * i / clusters)] = i
    return label

def value2int_simple(y):
    label = np.copy(y)
    label[y < 0] = 0
    label[y >= 0] = 1
    return label

def get_Train_Feature_Label(clusters=2, files=[], loc='', hasJunk=True):
    ##### train #####
    # data_files = [f for f in os.listdir(loc) if f.endswith('train.csv')]
    # data_files += [f for f in os.listdir(loc) if f.endswith('validation.csv')]
    data = pd.DataFrame()
    # for file in data_files[-150:]: # 只train部分
    for file in files: # train全部
        print(file)
        data = pd.concat([data, pd.read_csv(loc+file)])
    
    shuffle(data)
    X, y = data.iloc[:, 1:-1], data.iloc[:, -1]
    y = value2int_simple(y)

    epoch = 10
    print('Mean(y): ', np.mean(y))
    if np.mean(y) <= 0.55:
        epoch = 20

    # y = preprocessing.LabelEncoder().fit_transform(y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    # X, y = SMOTE().fit_resample(X, y)
    # X, y = TomekLinks().fit_resample(X, y)
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    # X_valid, y_valid = SMOTE().fit_resample(X_valid, y_valid)

    X_train, y_train = TomekLinks().fit_resample(X_train, y_train)
    # X_valid, y_valid = TomekLinks().fit_resample(X_valid, y_valid)
    print('Mean(y_train): ', np.mean(y_train))
    y_train = to_categorical(y_train).astype("int")
    y_valid = to_categorical(y_valid).astype("int")
    # print(X, y)
    # label = to_categorical(value2int_simple(y)).astype("int") # using direction to label
    #label = to_categorical(value2int(y, clusters)).astype("int") # using quantile to label
    # validation_ratio = 0.2
    X_train = X_train.values.reshape(X_train.shape[0], 30, 100, 1).astype('float32')
    X_valid = X_valid.values.reshape(X_valid.shape[0], 30, 100, 1).astype('float32')

    # D = int(data.shape[0] * validation_ratio)  # total number of validation data
    # X_train, y_train, X_valid, y_valid = X[:-D], label[:-D,:], X[-D:], label[-D:,:]

    # return X_train, y_train, X_valid, y_valid, epoch
    return X_train, y_train, X_valid, y_valid
    # np.save('./input2/X_train', X_train)
    # np.save('./input2/y_train', y_train)
    # np.save('./input2/X_valid', X_valid)
    # np.save('./input2/y_valid', y_valid)
    # del X_train, y_train, X_valid, y_valid

def get_Test_Feature_Label(clusters=2, files=[], hasJunk=True):
    ##### test #####
    if not os.path.exists('./input2/X_test.npy'):
        test = pd.DataFrame()
        for i in ['2021/', '2020/']:
            loc = './input2/stockFeatures_ForCNN/'
            loc += i
            files = sorted([f for f in os.listdir(loc) if f.endswith('test.csv')])
            # print(files)
            for file in files:
                print(file)
                test = pd.concat([test, pd.read_csv(loc+file)])
            # print(test.shape)
        X_test, y_test = test.iloc[:, 1:-1], test.iloc[:, -1]

        print("Positive News Ratio", sum(y_test > 0) * 1. / (sum(y_test > 0) + sum(y_test < 0)))
        X_test = X_test.values.reshape(X_test.shape[0], 30, 100, 1).astype('float32')
        y_test = to_categorical(value2int_simple(y_test)).astype("int")
        np.save('./input2/X_test', X_test)
        np.save('./input2/y_test', y_test)
        return X_test, y_test
    else:
        pass

def CNN(clusters=2):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 100), activation='relu', input_shape=(30, 100, 1))) # output channel, kernel size, input shape, input channel
    model.add(layers.MaxPooling2D((28, 1))) # 30-3+1 = 28 !!!
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(clusters, activation='softmax'))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    return model

def evaluate(model, clusters, X_train, y_train, X_valid, y_valid, X_test, y_test):
    model = keras.models.load_model('./input2/WB_CNN_Model.h5')
    
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=1024, verbose=2)
    # Final evaluation of the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)
    predictions = np.argmax(model.predict(X_valid), axis=-1)
    conf = confusion_matrix(np.argmax(y_valid, axis=-1), predictions)
    print(conf)
    for i in range(clusters):
        print("Valid Label %d Precision, %.2f%%" % (i, conf[i,i] * 100.0 / sum(conf[:,i])))
    print(f1_score(np.argmax(y_valid, axis=-1), predictions))
    # calculate predictions
    predictions = model.predict(X_test)
    # print(predictions)
    # print(y_test)
    fallThres = np.quantile(predictions[:, 0], 0.998)
    riseThres = np.quantile(predictions[:, 1], 0.998)

    for i in [0.5, 0.6, 0.65, 0.7, 0.75]:
        try:
            thres = i
            print('\nThreshold:', i)
            y_cut = (predictions[:,0] > thres) | (predictions[:,1] > thres) # cut y value and leave the better result
            predictions_2 = np.argmax(predictions[y_cut], axis=-1)
            conf = confusion_matrix(np.argmax(y_test[y_cut], axis=-1), predictions_2)
            print("Test on %d samples" % (len(y_test[y_cut])))
            print(conf)
            for i in range(clusters):
                print("Test Label %d Precision, %.2f%%" % (i, conf[i,i] * 100.0 / sum(conf[:,i])))
            print(f1_score(np.argmax(y_test[y_cut], axis=-1), predictions_2))
        except Exception as e:
            print(e)

    print('FallThreshold:', round(fallThres, 3), 'RiseThreshold', round(riseThres, 3))
    y_cut = (predictions[:,0] > fallThres) | (predictions[:,1] > riseThres) # cut y value and leave the better result
    predictions_2 = np.argmax(predictions[y_cut], axis=-1)
    conf = confusion_matrix(np.argmax(y_test[y_cut], axis=-1), predictions_2)
    print("Test on %d samples" % (len(y_test[y_cut])))
    print(conf)
    for i in range(clusters):
        print("Test Label %d Precision, %.2f%%" % (i, conf[i,i] * 100.0 / sum(conf[:,i])))

    print(f1_score(np.argmax(y_test[y_cut], axis=-1), predictions_2))
    model.save('./input2/WB_CNN_Model.h5', save_format="h5")

def model_selection(clusters): # random sampling is better than grid search
    # try:
    #     X_train = np.load('./input2/X_train.npy')
    #     y_train = np.load('./input2/y_train.npy')
    #     X_valid = np.load('./input2/X_valid.npy')
    #     y_valid = np.load('./input2/y_valid.npy')
    #     X_test = np.load('./input2/X_test.npy')
    #     y_test = np.load('./input2/y_test.npy')
    #     print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)
    # except:
    #     get_Feature_Label(clusters=clusters)
    #     X_train = np.load('./input2/X_train.npy')
    #     y_train = np.load('./input2/y_train.npy')
    #     X_valid = np.load('./input2/X_valid.npy')
    #     y_valid = np.load('./input2/y_valid.npy')
    #     X_test = np.load('./input2/X_test.npy')
    #     y_test = np.load('./input2/y_test.npy')
    try:
        X_test = np.load('./input2/X_test.npy')
        y_test = np.load('./input2/y_test.npy')
    except:
        # loc = './input2/stockFeatures_ForCNN/'
        # data_files = sorted([f for f in os.listdir(loc) if f.endswith('test.csv')])
        X_test, y_test = get_Test_Feature_Label(clusters=2)
    # print(len(y_test))
    loc = './input2/stockFeatures_ForCNN/'
    model = CNN(clusters)
    model.save('./input2/WB_CNN_Model.h5', save_format="h5")
    

    Epoch = 5
    for epoch in range(Epoch):
        print('Epoch: ', epoch)

        for year in range(2013, 2020):
        # for year in [2019]:
            loc = './input2/stockFeatures_ForCNN/'
            loc = loc + str(year) + '/'
            data_files = sorted([f for f in os.listdir(loc) if f.endswith('train.csv')])
            # print(len(data_files)) # 815 => 81
            for i in range(len(data_files)//10): ### i = 0~80
                print('Epoch: ', epoch, 'Year:', year, 'Train Set: ', i)
                if i != (len(data_files)//10 - 1):
                    files = data_files[i*10 : (i+1)*10]
                    # files = data_files[0:2]
                else:
                    files = data_files[i*10 :          ]
                X_train, y_train, X_valid, y_valid = get_Train_Feature_Label(clusters=clusters, files=files, loc=loc)
                # X_train = np.load('./input2/X_train.npy')
                # y_train = np.load('./input2/y_train.npy')
                # X_valid = np.load('./input2/X_valid.npy')
                # y_valid = np.load('./input2/y_valid.npy')

                evaluate(model, clusters, X_train, y_train, X_valid, y_valid, X_test, y_test)
                
                del X_train, y_train, X_valid, y_valid

def main():
    clusters = 2
    model_selection(clusters)
    


if __name__ == "__main__":
    main()
    set_keepawake(keep_screen_awake=True)

