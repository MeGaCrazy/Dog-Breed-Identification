# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:02:50 2018

@author: CS
"""


from __future__ import print_function
import numpy as np
from codes.Preprocess import *

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,CSVLogger
import tensorflow as tf 
from keras import backend as k

def categorical_hinge(y_true, y_pred):
    pos = k.sum(y_true * y_pred, axis=-1)
    neg =k.max((1.0 - y_true) * y_pred, axis=-1)
    return k.mean(k.maximum(0.0, neg - pos + 1), axis=-1)


# Data Preparing
X, y, our_classes = data_reader()
X, y = shuffle(X, y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)

batch_size = 4
nr_classes = 10
nr_iterations = 10
#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)

X_train /= 255
X_test /= 255


Y_train = np_utils.to_categorical(y_train, nr_classes)
Y_test = np_utils.to_categorical(y_test, nr_classes)

input_shape=(28,28,1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))




X_val=X_train[500:1000,:]
Y_val=Y_train[500:1000,:]


model.summary()
model.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])

saved_weights_name='SVMWeights.h5'

checkpoint = ModelCheckpoint(saved_weights_name, 
                                     monitor='val_acc', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='max')
csv_logger = CSVLogger('v.csv')

history = model.fit(X_train, Y_train,
                    batch_size = batch_size, nb_epoch = nr_iterations,
                    verbose = 1, validation_data = (X_val, Y_val) ,callbacks=[checkpoint,csv_logger])

score = model.evaluate(X_test, Y_test, verbose = 0)