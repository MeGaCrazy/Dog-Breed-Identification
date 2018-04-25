import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt



def Most_freq():
    labels = pd.read_csv("..\labels.csv")
    class_frequency = labels.breed.value_counts()
    print("The Most Frequency classes in The DataSet:\nBreed                   Freq")
    print(class_frequency.head())
    ret_Most_freq_dog = class_frequency.head()
    return ret_Most_freq_dog.keys()


def Prepare5class(classes):
    print('Getting Small data Ready...')
    df = pd.read_csv("..\labels.csv")
    for idx, row in df.iterrows():
        if (str(row['breed'])) in classes:
            image_path = "..\Train/" + row['id'] + '.jpg'
            image_new_path = "..\smallTrain/" + row['id'] + '.jpg'
            os.rename(image_path, image_new_path)
    print('Small Data is Ready.')


if __name__ == '__main__':
    ret = Most_freq()
    Prepare5class(ret)
