import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder


def top5class():
    labels = pd.read_csv("..\labels.csv")
    class_frequency = labels.breed.value_counts()
    ret_Most_freq_dog = class_frequency.head(n=10)
    return ret_Most_freq_dog.keys()


def data_reader(cnn):
    data = []
    labels = []
    print('Reading Data..')
    df = pd.read_csv("..\labels.csv")
    print('resizing and preparing the images:')
    our_classes = top5class()
    for idx, row in df.iterrows():
        if str(row['breed']) in our_classes:
            image_path = "..\smallTrain/" + row['id'] + '.jpg'
            img = cv2.imread(image_path)
            img = cv2.resize(img, (80, 80))
            img = img.reshape([80, 80, 3])
            cur_label = our_classes.get_loc(str(row['breed']))
            labels.append(cur_label)
            data.append(img)
    img_data = np.array(data)
    img_data = img_data.astype('float32')
    if cnn == 0:
        img_data = img_data.flatten().reshape(img_data.shape[0], 80 * 80 * 3)
    label_data = np.array(labels)
    print('The Data is Ready.')
    print('Number of m : ' + str(img_data.shape[0]) + " <<<<<>>>>> " + ' Number of Features is: ' + str(
        img_data.shape[1]))
    return img_data, label_data, our_classes
