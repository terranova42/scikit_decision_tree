#!/usr/bin/env python
# -*- coding: utf-8 -*-

# "hello_world" classifier
# source: https://www.youtube.com/watch?v=cKxRvEZd3Mw

from sklearn import tree
import numpy as np
import os
import sys
import cv2
import re


datadir = "./apple_orange_images/"
data_type = "fruits"

# features = data set characteristics
# labels = type of data

def test():
    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = [1, 1, 0, 0]

    # choose algo for classification
    clf = tree.DecisionTreeClassifier()
    # train
    clf = clf.fit(features, labels)
    # predict
    print(clf.predict([[160, 0]]))

def search_label_in_filename(str):
    if data_type == "orange_apple":
        apple = re.compile('apple', re.IGNORECASE)
        orange = re.compile('orange', re.IGNORECASE)

        if orange.search(str):
            return 2
        elif apple.search(str):
            return 1
        else:
            return 0

    elif data_type == "porsche_mercedes":
        porsche = re.compile('porsche', re.IGNORECASE)
        mercedes = re.compile('mercedes', re.IGNORECASE)

        if porsche.search(str):
            return 2
        elif mercedes.search(str):
            return 1
        else:
            return 0

    else:
        return None

def num_to_label(num):
    if data_type == "orange_apple":
        if num == 2:
            return "orange"
        elif num == 1:
            return "apple"
        else:
            return "null"
    elif data_type == "porsche_mercedes":
        if num == 2:
            return "porsche"
        elif num == 1:
            return "mercedes"
        else:
            return "null"
    else:
        return None

def info_data(features,labels):
    print("imported features: len{}, type: {}, with labels len={}, type={}".format(len(features), type(features), len(labels), type(labels)))
    print(labels)

    
def resize_images_to_highest_dim(img_array):
    #takes in input an array of images and resized all images to the highest dimension

    height=0
    width=0

    for img in img_array:
        shape = np.shape(img)
        if height < shape[0]:
            height = shape[0]
        if width < shape[1]:
            width = shape[1]

    resized_array = []
    for img in img_array:
        resized_img = cv2.resize(img,(width,height))
        resized_array.append(resized_img)

    num_img = 0
    for filename in os.listdir(datadir):
        cv2.imwrite(datadir+'resized_'+filename, resized_array[num_img])
        num_img += 1

    return resized_array

def flatten_array(img_array):

    flat_img_array = []

    for img in img_array:
        flat_img = np.concatenate(img).ravel()
        flat_img_array.append(flat_img)

    return flat_img_array


def image_classifier(file):

    '''
    :param file: a file to classify, 'apple' or 'orange'
    :return: 'apple' or 'orange' keyword

    ->foreach file in current dir:
    import file with opencv
    add file (numpy array format) to features array
    add label 'orange' or 'apple' depending on the filename

    -> train

    -> predict input file

    '''
    features = []
    labels = []

    for img in os.listdir(datadir):
        np_array = cv2.imread(datadir + img)
        #flat_array = np.concatenate(np_array).ravel()
        features.append(np_array)
        labels.append(search_label_in_filename(img))

    #info_data(features, labels)

    #  choose algo for classification
    clf = tree.DecisionTreeClassifier()

    # resize all features to the highest lenght
    resized_features = resize_images_to_highest_dim(features)
    flat_features = flatten_array(resized_features)

    #  train
    clf = clf.fit(flat_features, labels)

    #  predict
    file_numpy = cv2.imread(file)

    target_shape = resized_features[0].shape
    target_height = target_shape[0]
    target_width = target_shape[1]

    # resize input image
    resized_input = cv2.resize(file_numpy,(target_width,target_height))
    #flatten to 1d array
    flat_input = np.concatenate(resized_input).ravel()

    normalized_array = np.reshape(flat_input, (1,-1))
    result = clf.predict(normalized_array)
    proba = clf.predict_proba(normalized_array)
    #print(num_to_label(result,"orange_apple"))
    print("image is recognized as: {} with a probabilty of: {}".format(num_to_label(result),proba))

def usage():
    print("usage: {} data_type filename_to_predict\n first argument data_type = 'fruit' or 'cars', second argument is the image to be guessed.".format(sys.argv[0]))
    exit(1)

def main():
    global data_type
    global datadir
    if len(sys.argv) != 3:
        usage()
    else:
        if sys.argv[1] == "fruit":
            data_type="orange_apple"
            datadir="./apple_orange_images/"
        elif sys.argv[1] == "car":
            data_type="porsche_mercedes"
            datadir="./porsche_mercedes/"
        else:
            usage()

        image_classifier(sys.argv[2])
        exit(0)


if __name__ == "__main__":
    #test()
    main()

'''
example X = features

>>> X
array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       [5. , 3.6, 1.4, 0.2],
       [5.4, 3.9, 1.7, 0.4],
       [4.6, 3.4, 1.4, 0.3],
       [5. , 3.4, 1.5, 0.2],
       [4.4, 2.9, 1.4, 0.2],
       [4.9, 3.1, 1.5, 0.1],
       [5.4, 3.7, 1.5, 0.2],
       [4.8, 3.4, 1.6, 0.2],
       [4.8, 3. , 1.4, 0.1],
       [4.3, 3. , 1.1, 0.1],
       [5.8, 4. , 1.2, 0.2],
       [5.7, 4.4, 1.5, 0.4],
       [5.4, 3.9, 1.3, 0.4],
       [5.1, 3.5, 1.4, 0.3],
       [5.7, 3.8, 1.7, 0.3],
       [5.1, 3.8, 1.5, 0.3],
       [5.4, 3.4, 1.7, 0.2],
       [5.1, 3.7, 1.5, 0.4],
       [4.6, 3.6, 1. , 0.2],
       [5.1, 3.3, 1.7, 0.5],
       [4.8, 3.4, 1.9, 0.2],
       [5. , 3. , 1.6, 0.2],
       [5. , 3.4, 1.6, 0.4],
       [5.2, 3.5, 1.5, 0.2],
       [5.2, 3.4, 1.4, 0.2],
       [4.7, 3.2, 1.6, 0.2],
       [4.8, 3.1, 1.6, 0.2],
       [5.4, 3.4, 1.5, 0.4],
       [5.2, 4.1, 1.5, 0.1],
       [5.5, 4.2, 1.4, 0.2],
       [4.9, 3.1, 1.5, 0.2],
       [5. , 3.2, 1.2, 0.2],
       [5.5, 3.5, 1.3, 0.2],
       [4.9, 3.6, 1.4, 0.1],
       [4.4, 3. , 1.3, 0.2],
       [5.1, 3.4, 1.5, 0.2],
       [5. , 3.5, 1.3, 0.3],
       [4.5, 2.3, 1.3, 0.3],
       [4.4, 3.2, 1.3, 0.2],
       [5. , 3.5, 1.6, 0.6],
       [5.1, 3.8, 1.9, 0.4],
       [4.8, 3. , 1.4, 0.3],
       [5.1, 3.8, 1.6, 0.2],
       [4.6, 3.2, 1.4, 0.2],
       [5.3, 3.7, 1.5, 0.2],
       [5. , 3.3, 1.4, 0.2],
       [7. , 3.2, 4.7, 1.4],
       [6.4, 3.2, 4.5, 1.5],
       [6.9, 3.1, 4.9, 1.5],
       [5.5, 2.3, 4. , 1.3],
       [6.5, 2.8, 4.6, 1.5],
       [5.7, 2.8, 4.5, 1.3],
       [6.3, 3.3, 4.7, 1.6],
       [4.9, 2.4, 3.3, 1. ],
       [6.6, 2.9, 4.6, 1.3],
       [5.2, 2.7, 3.9, 1.4],
       [5. , 2. , 3.5, 1. ],
       [5.9, 3. , 4.2, 1.5],
       [6. , 2.2, 4. , 1. ],
       [6.1, 2.9, 4.7, 1.4],
       [5.6, 2.9, 3.6, 1.3],
       [6.7, 3.1, 4.4, 1.4],
       [5.6, 3. , 4.5, 1.5],
       [5.8, 2.7, 4.1, 1. ],
       [6.2, 2.2, 4.5, 1.5],
       [5.6, 2.5, 3.9, 1.1],
       [5.9, 3.2, 4.8, 1.8],
       [6.1, 2.8, 4. , 1.3],
       [6.3, 2.5, 4.9, 1.5],
       [6.1, 2.8, 4.7, 1.2],
       [6.4, 2.9, 4.3, 1.3],
       [6.6, 3. , 4.4, 1.4],
       [6.8, 2.8, 4.8, 1.4],
       [6.7, 3. , 5. , 1.7],
       [6. , 2.9, 4.5, 1.5],
       [5.7, 2.6, 3.5, 1. ],
       [5.5, 2.4, 3.8, 1.1],
       [5.5, 2.4, 3.7, 1. ],
       [5.8, 2.7, 3.9, 1.2],
       [6. , 2.7, 5.1, 1.6],
       [5.4, 3. , 4.5, 1.5],
       [6. , 3.4, 4.5, 1.6],
       [6.7, 3.1, 4.7, 1.5],
       [6.3, 2.3, 4.4, 1.3],
       [5.6, 3. , 4.1, 1.3],
       [5.5, 2.5, 4. , 1.3],
       [5.5, 2.6, 4.4, 1.2],
       [6.1, 3. , 4.6, 1.4],
       [5.8, 2.6, 4. , 1.2],
       [5. , 2.3, 3.3, 1. ],
       [5.6, 2.7, 4.2, 1.3],
       [5.7, 3. , 4.2, 1.2],
       [5.7, 2.9, 4.2, 1.3],
       [6.2, 2.9, 4.3, 1.3],
       [5.1, 2.5, 3. , 1.1],
       [5.7, 2.8, 4.1, 1.3],
       [6.3, 3.3, 6. , 2.5],
       [5.8, 2.7, 5.1, 1.9],
       [7.1, 3. , 5.9, 2.1],
       [6.3, 2.9, 5.6, 1.8],
       [6.5, 3. , 5.8, 2.2],
       [7.6, 3. , 6.6, 2.1],
       [4.9, 2.5, 4.5, 1.7],
       [7.3, 2.9, 6.3, 1.8],
       [6.7, 2.5, 5.8, 1.8],
       [7.2, 3.6, 6.1, 2.5],
       [6.5, 3.2, 5.1, 2. ],
       [6.4, 2.7, 5.3, 1.9],
       [6.8, 3. , 5.5, 2.1],
       [5.7, 2.5, 5. , 2. ],
       [5.8, 2.8, 5.1, 2.4],
       [6.4, 3.2, 5.3, 2.3],
       [6.5, 3. , 5.5, 1.8],
       [7.7, 3.8, 6.7, 2.2],
       [7.7, 2.6, 6.9, 2.3],
       [6. , 2.2, 5. , 1.5],
       [6.9, 3.2, 5.7, 2.3],
       [5.6, 2.8, 4.9, 2. ],
       [7.7, 2.8, 6.7, 2. ],
       [6.3, 2.7, 4.9, 1.8],
       [6.7, 3.3, 5.7, 2.1],
       [7.2, 3.2, 6. , 1.8],
       [6.2, 2.8, 4.8, 1.8],
       [6.1, 3. , 4.9, 1.8],
       [6.4, 2.8, 5.6, 2.1],
       [7.2, 3. , 5.8, 1.6],
       [7.4, 2.8, 6.1, 1.9],
       [7.9, 3.8, 6.4, 2. ],
       [6.4, 2.8, 5.6, 2.2],
       [6.3, 2.8, 5.1, 1.5],
       [6.1, 2.6, 5.6, 1.4],
       [7.7, 3. , 6.1, 2.3],
       [6.3, 3.4, 5.6, 2.4],
       [6.4, 3.1, 5.5, 1.8],
       [6. , 3. , 4.8, 1.8],
       [6.9, 3.1, 5.4, 2.1],
       [6.7, 3.1, 5.6, 2.4],
       [6.9, 3.1, 5.1, 2.3],
       [5.8, 2.7, 5.1, 1.9],
       [6.8, 3.2, 5.9, 2.3],
       [6.7, 3.3, 5.7, 2.5],
       [6.7, 3. , 5.2, 2.3],
       [6.3, 2.5, 5. , 1.9],
       [6.5, 3. , 5.2, 2. ],
       [6.2, 3.4, 5.4, 2.3],
       [5.9, 3. , 5.1, 1.8]])
>>> type(X)
<class 'numpy.ndarray'>


example y = labels

>>> y
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
>>> type(y)
<class 'numpy.ndarray'>



'''




