# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:02:50 2018

@author: CS
"""

from __future__ import print_function
import numpy as np
from codes.Preprocess import *

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization, Input, Dense, Convolution2D, MaxPooling2D, \
    AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger
import warnings
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

warnings.filterwarnings('ignore')

# Data Preparing

batch_size = 4
nr_classes = 10
nr_iterations = 10
X, y, our_classes = data_reader(1)
X, y = shuffle(X, y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nr_classes)
Y_test = np_utils.to_categorical(y_test, nr_classes)

X_val = X_train[500:1000, :]
Y_val = Y_train[500:1000, :]

X_train = X_train[0:500, :]
Y_train = Y_train[0:500, :]

input_shape = (80, 80, 3)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
# saved_weights_name='SVMWeights.h5'
# model.load_weights(saved_weights_name)
model.summary()

x = model.layers[4].output

x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='sigmoid')(x)

model = Model(inputs=model.input, outputs=predictions)

for layer in model.layers[0:4]:
    layer.trainable = False

model.summary()
model.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])
saved_weights_name = 'CNNWeights.h5'
checkpoint = ModelCheckpoint(saved_weights_name,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
csv_logger = CSVLogger('CNN.csv')

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nr_iterations,
                    verbose=1, validation_data=(X_val, Y_val), callbacks=[checkpoint, csv_logger])

score = model.evaluate(X_test, Y_test, verbose=0)
