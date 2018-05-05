from __future__ import print_function
import numpy as np
from codes.Preprocess import *

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf
from keras import backend as k
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


def categorical_hinge(y_true, y_pred):
    pos = k.sum(y_true * y_pred, axis=-1)
    neg = k.max((1.0 - y_true) * y_pred, axis=-1)
    return k.mean(k.maximum(0.0, neg - pos + 1), axis=-1)


# Data Preparing

batch_size = 128
nr_classes = 10
nr_iterations = 10
X, y, our_classes = data_reader(0)
X, y = shuffle(X, y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nr_classes)
Y_test = np_utils.to_categorical(y_test, nr_classes)

model = Sequential()
model.add(Dense(3000, input_shape=(80*80*3,)))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

X_val = X_train[500:1000, :]
Y_val = Y_train[500:1000, :]
#
model.summary()
model.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])

saved_weights_name = 'SVMWeights.h5'

checkpoint = ModelCheckpoint(saved_weights_name,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
csv_logger = CSVLogger('SVM.csv')

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nr_iterations,
                    verbose=1, validation_data=(X_val, Y_val), callbacks=[checkpoint, csv_logger])

score = model.evaluate(X_test, Y_test, verbose=0)
