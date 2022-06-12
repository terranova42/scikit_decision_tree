#!/usr/bin/env python
# -*- coding: utf-8 -*-

# "hello_world" classifier
# source: https://www.youtube.com/watch?v=cKxRvEZd3Mw

# V.1.2: added "save to disk" with pickle

# V.1.3: 11 june 2022:tested with SGD Classifier.
# all images are properly scaled to best dimension.
# todo: check performance with python on windows 10. memory limits.
# todo: with SGD, check data normalization: try without normalization (no flattening arrays),
# todo: check built-in features scaling with SGD.
# todo: check online tutorials for best practices.
# test with other models: SVM, ...
# learn how to optimize.

from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import sys
import cv2
import re
import pickle

data_dir = "./fruits_train/"
test_dir = "./fruits_test/"
data_type = "fruits"
# classifier = "DecisionTreeClassifier"
classifier = "SGD"


# features = data set characteristics
# labels = type of data

def hello_world():
    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = [1, 1, 0, 0]

    # choose algo for classification
    clf = tree.DecisionTreeClassifier()
    # train
    clf = clf.fit(features, labels)
    # predict
    print(clf.predict([[160, 0]]))


def search_label_in_filename(str):
    if data_type == "fruits":
        apple = re.compile('apple', re.IGNORECASE)
        orange = re.compile('orange', re.IGNORECASE)

        if orange.search(str):
            return 2
        elif apple.search(str):
            return 1
        else:
            print("alert! no label found in file: {}".format(str))
            return 0

    elif data_type == "cars":
        porsche = re.compile('porsche', re.IGNORECASE)
        mercedes = re.compile('mercedes', re.IGNORECASE)

        if porsche.search(str):
            return 2
        elif mercedes.search(str):
            return 1
        else:
            print("alert! no label found in file: {}".format(str))
            return 0

    else:
        return None


def num_to_label(num):
    if data_type == "fruits":
        if num == 2:
            return "orange"
        elif num == 1:
            return "apple"
        else:
            return "null"
    elif data_type == "cars":
        if num == 2:
            return "porsche"
        elif num == 1:
            return "mercedes"
        else:
            return "null"
    else:
        return None


def info_data(features, labels):
    print("imported features: len{}, type: {}, with labels len={}, type={}".format(len(features), type(features),
                                                                                   len(labels), type(labels)))
    print(labels)


def find_heighest_height_and_width(img_array1, img_array2):
    # takes in input arrays of images and return the highest dimensions

    height = 0
    width = 0

    for img in img_array1:
        shape = np.shape(img)
        if height < shape[0]:
            height = shape[0]
        if width < shape[1]:
            width = shape[1]

    for img in img_array2:
        shape = np.shape(img)
        if height < shape[0]:
            height = shape[0]
        if width < shape[1]:
            width = shape[1]

    return height, width


def resize_images_to_best_dim(img_array, height, width):
    resized_array = []
    for img in img_array:
        resized_img = cv2.resize(img, (width, height), 0, 0, cv2.INTER_LINEAR)
        resized_array.append(resized_img)

    '''
    num_img = 0
    for filename in os.listdir(data_dir):
        cv2.imwrite(data_dir+'resized_'+filename, resized_array[num_img])
        num_img += 1
    '''
    return resized_array



def flatten_array(img_array):
    flat_img_array = []

    for img in img_array:
        # flat_img = np.concatenate(img).ravel()
        # print("img.shape = {}".format(img.shape)) #3-dim
        flat_img = img.reshape(-1)
        # print("flat_img.shape = {}".format(flat_img.shape)) #1-dim
        # print("flat_img size = {}".format(len(flat_img)))
        flat_img_array.append(flat_img)

    return flat_img_array


def save_model(modname):
    s = pickle.dumps(modname)
    filename = classifier + "_" + data_type + "_trained_model.pickle"
    with open(filename, 'wb') as f:
        f.write(s)
        f.close()


def check_model_on_disk():
    filename = classifier + "_" + data_type + "_trained_model.pickle"
    if os.access(filename, os.R_OK):
        with open(filename, 'rb') as f:
            s = f.read()
            model = pickle.loads(s)
            return model
    else:
        return None


def predict_target_features(target_array, filenames, model):

    n = 0
    for n in range(len(target_array)):
        resized_array = target_array[n].reshape(1, -1)
        result = model.predict(resized_array)
        # proba = model.predict_proba(normalized_array)
        # print(num_to_label(result,"orange_apple"))
        # print("image {} is recognized as: {} with a probability of: {}".format(file, num_to_label(result), proba))
        print("image {} is recognized as: {}".format(filenames[n], num_to_label(result)))


def get_array_of_target_features(file):
    # returns an array with the target file to predict: either the 1 file on command line,
    # or the files into test_directory.

    # should implement a map/dictionnary/numpy structure
    my_filenames = []
    my_target_features = []
    if file == "TEST":
        for f in os.listdir(test_dir):
            # print(f)
            np_array = cv2.imread(test_dir + f)
            my_target_features.append(np_array)
            my_filenames.append(f)
    else:
        np_array = cv2.imread(file)
        # imread returns an np array
        # print("type of np_array = {}".format(type(np_array)))
        my_target_features.append(np_array)
        my_filenames.append(file)

    # print(my_filenames)
    return my_target_features, my_filenames

def display_images(img_array):
    for img in img_array:
        cv2.namedWindow('test image', cv2.WINDOW_NORMAL)
        cv2.imshow("test image", img)
        cv2.waitKey()

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

    for img in os.listdir(data_dir):
        np_array = cv2.imread(data_dir + img)
        # flat_array = np.concatenate(np_array).ravel()
        features.append(np_array)
        img_class = search_label_in_filename(img)
        labels.append(img_class)
        # print("image: {} labelled as {}".format(img, img_class))

    # info_data(features, labels)

    # choose algo for classification
    # model = tree.DecisionTreeClassifier()
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    # classifier = "SGD"

    # resize all features to the highest lenght
    target_features, target_filenames = get_array_of_target_features(file)
    best_height, best_width = find_heighest_height_and_width(features, target_features)
    # print(best_height)
    # print(best_width)
    # resize train features
    resized_features = resize_images_to_best_dim(features, best_height, best_width)
    flat_features = flatten_array(resized_features)
    # resize test features
    resized_target_features = resize_images_to_best_dim(target_features, best_height, best_width)
    flat_target_features = flatten_array(resized_target_features)

    # print(len(flat_features))
    # display images:
    # display_images(resized_target_features)

    # exit(1)
    # Stochastic Gradient Descent is sensitive to feature scaling
    # scaler = StandardScaler()

    # check trained model on disk, else train a new one.
    saved = check_model_on_disk()
    if saved is not None:
        model = saved
        print("loaded saved model...")
    else:
        # train
        print("training new model...")
        # model = model.fit(flat_features, labels)
        # scale training data
        # scaler.fit(flat_features)
        # flat_features = scaler.transform(flat_features)

        model = model.fit(flat_features, labels)
        # save model on disk with pickle
        save_model(model)
        print("saved model on disk...")

    # target_shape = resized_features[0].shape
    # target_height = target_shape[0]
    # target_width = target_shape[1]

    # scale test data
    # flat_target_features = scaler.transform(flat_target_features)

    # predict
    predict_target_features(flat_target_features, target_filenames, model)

    # file_numpy = cv2.imread(file)

    # resize input image
    # resized_input = cv2.resize(file_numpy, (target_width, target_height))
    # flatten to 1d array
    # flat_input = np.concatenate(resized_input).ravel()
    # normalized_array = np.reshape(flat_input, (1, -1))
    # result = model.predict(normalized_array)
    # proba = model.predict_proba(normalized_array)
    # print(num_to_label(result,"orange_apple"))
    # print("image {} is recognized as: {} with a probability of: {}".format(file,num_to_label(result),proba))
    # print("image {} is recognized as: {}".format(file, num_to_label(result)))


def usage():
    print("usage: {} data_type filename_to_predict\nfirst argument data_type = 'fruits' or 'cars', "
          "second argument is the image to be guessed. \n"
          "TEST special keyword will test all images in test dir.".format(sys.argv[0]))
    exit(1)


def main():
    global data_type
    global data_dir
    global test_dir
    if len(sys.argv) != 3:
        usage()
    else:
        if sys.argv[1] == "fruits":
            data_type = "fruits"
            data_dir = "./fruits_train/"
            test_dir = "./fruits_test/"
        elif sys.argv[1] == "cars":
            data_type = "cars"
            data_dir = "./cars_train/"
            test_dir = "./cars_test/"
        else:
            usage()

        if not os.access(sys.argv[2], os.R_OK):
            if sys.argv[2] != "TEST":
                print("wrong filename or wrong permission: {}".format(sys.argv[2]))
                exit(1)

        image_classifier(sys.argv[2])
        exit(0)


if __name__ == "__main__":
    # hello_world()
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
