from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import read_data

def main():
    image_model = train_image_model()

def train_biz_model(image_model):
    # need pic -> biz mapping
    # for each biz, find pic

    # train data: vector size 9, number of images that predict that label
    pass 

def train_image_model():
    batch_size = 128
    nb_classes = 9
    nb_epoch = 3
    dropout = 0.2

    # input image dimensions
    img_size = 200
    num_biz = 100
    # number of convolutional filters to use
    nb_filters = 30
    # size of pooling area for max pooling
    nb_pool = 10 
    # convolution kernel size
    nb_conv = 4

    # the data, shuffled and split between train and test sets
    (X_train, Y_train, X_test, Y_test) = \
        read_data.read_data_photo_labels(img_size = img_size, num_biz = num_biz)

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('Training on %s images, testing on %s images' % \
            (X_train.shape[0], X_test.shape[0]))

  # datagen = ImageDataGenerator(
  #     featurewise_center=False,
  #     featurewise_std_normalization=False,
  #     rotation_range=20,
  #     zca_whitening=False,
  #     width_shift_range=0.2,
  #     height_shift_range=0.2,
  #     horizontal_flip=True)
  # datagen.fit(X_train)

    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(3, img_size, img_size)))
    model.add(Activation('relu'))

  # model.add(MaxPooling2D(pool_size=(3,3)))
  # model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
  # model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(900))
    model.add(Activation('relu'))

    model.add(Reshape((1, 30, 30)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1024))
    model.add(Activation('relu'))

 #  model.add(Dropout(dropout))
 #  model.add(Dense(128))
 #  model.add(Activation('relu'))

    model.add(Dropout(dropout))
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(nb_classes))

    model.compile(loss='hinge', optimizer='adadelta')

    model.fit(
            X_train, Y_train, batch_size=32,
            nb_epoch=nb_epoch, verbose=1)

 #  model.fit_generator(
 #          datagen.flow(X_train, Y_train, batch_size=32), 
 #          samples_per_epoch=18226, nb_epoch=nb_epoch, verbose=1)
    test_pred = np.sign(model.predict(X_test))
    test_loss = model.evaluate(X_test, Y_test)
    np.savetxt('pred.csv', test_pred, delimiter=',')

    print('Test loss: ', test_loss)
    print('Test accuracy: ', accuracy(test_pred, Y_test))
    print('F1 score: ', f1score(test_pred, Y_test))

    return model

def accuracy(y_pred, y_test):
    total = y_test.size
    correct = np.sum(y_pred == y_test)
    return correct/total

def f1score(y_pred, y_test):
    tp = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_test == -1))
    fn = np.sum(np.logical_and(y_pred == -1, y_test == 1))
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    return 2*precision*recall/(precision + recall)

main()
