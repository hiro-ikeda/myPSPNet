from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

import numpy as np
import pandas as pd
from PIL import Image
import random
import matplotlib.pyplot as plt


"""
convert mask image's array(height, width, 1) to num_class dimension image's array(height, width, num_class)
"""
def binarylab(labels, img_size, nb_class) :
    y = np.zeros(( img_size[0], img_size[1], nb_class))
    for i in range(img_size[0]) :
        for j in range(img_size[1]) :
            y[i, j, labels[i][j]] = 1

    return y

"""
load data from directory
"""
#load image's data and covert image to array and apply some preprocess
def load_img_data(path, img_size, mode=None, nb_class=None) :

    img = Image.open(path)

    if mode == "data" :

        X = img_to_array(img)
        X = X / 255.0
        X = np.expand_dims(X, axis=0)
        #import pdb; pdb.set_trace()
        X = preprocess_input(X)

        return X

    elif mode == "label" :

        y = np.array(img, dtype=np.int32)

        y = binarylab(y, img_size, nb_class)
        y = np.expand_dims(y, axis=0)

        return y

"""
generator(create tupple including iamge and label)
"""
#load input data from jpeg file and load label data from png
# - call preprocess method
def generate_arrays_from_images(names, path_to_train, path_to_label, img_size, nb_class) :
    while True :
        for name in names :
            XPath = path_to_train + "{}.jpg".format(name)
            ypath = path_to_label + "{}.png".format(name)
            X = load_img_data(XPath, img_size, mode="data")
            y = load_img_data(ypath, img_size, mode="label", nb_class=nb_class)

            yield(X, y)

"""
group imags and labels by batch
"""
#in case no datra augmentation
def group_by_batch (dataset, batchsize) :

    while True :
        datas, labels = zip(*[next(dataset) for i in range(batchsize)])

        batch = (np.concatenate(datas, axis=0), np.concatenate(labels, axis=0))

        yield batch

"""
load data,labels from directory and group images,labels by batch and return to fit_generator(train.py)
"""
def load_dataset (names, path_to_train, path_to_label, img_size, nb_class, num_batchsizes):

    generator = generate_arrays_from_images(names, path_to_train, path_to_label, img_size, nb_class)

    batch = group_by_batch(generator, num_batchsizes)

    return batch
