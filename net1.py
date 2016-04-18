from __future__ import print_function
import numpy as np
import sys
np.random.seed(1307)  # for reproducibility

from keras import models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import read_data
import score

JSON_NAME = 'net1.json'
WEIGHTS_NAME = 'net1.h5'

def main():
    batch_size = 64
    nb_epoch = 20
    img_size = 256
    gen = read_data.read_data_photo_labels(
            2000, img_size, batch_size = batch_size)
    X_test, Y_test = next(gen)

  # model = get_image_model(img_size)

    with open(JSON_NAME, 'r') as jfile:
        model = models.model_from_json(jfile.read())
    model.load_weights(WEIGHTS_NAME)

  # model.fit_generator(gen, 
  #         samples_per_epoch=10000, nb_epoch=nb_epoch, verbose=1)

    test_pred = np.sign(model.predict(X_test))
    test_loss = model.evaluate(X_test, Y_test)
    np.savetxt('pred.csv', test_pred, delimiter=',')

  # with open(JSON_NAME, 'w') as jfile:
  #     jfile.write(model.to_json())
  # model.save_weights(WEIGHTS_NAME, overwrite=True)

    print('Test loss: ', test_loss)
    print('Test accuracy: ', score.accuracy(test_pred, Y_test))
    print('F1 score: ', score.f1score(test_pred, Y_test))
    print('F1 score by class:')
    score_byclass = score.f1_by_class(test_pred, Y_test)
    for c, s in enumerate(score_byclass):
        print(c, ':', s)

def train_biz_model(image_model):
    # need pic -> biz mapping
    # for each biz, find pic

    # train data: vector size 9, number of images that predict that label
    pass 

def get_image_model(img_size):

    nb_classes = 9
    dropout = 0.5
    # input image dimensions
    # number of convolutional filters to use
    nb_filters = 20
    # size of pooling area for max pooling
    nb_pool = 4 
    # convolution kernel size
    nb_conv = 4

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

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

  # model.add(Flatten())
  # model.add(Dropout(dropout))
  # model.add(Dense(900))
  # model.add(Activation('relu'))

  # model.add(Reshape((1, 30, 30)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
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

    model.compile(loss='hinge', optimizer='adam')

    # the data, shuffled and split between train and test sets
 #  (X_train, Y_train, X_test, Y_test) = \
 #      read_data.read_data_photo_labels(img_size = img_size, num_biz = num_biz)

 #  print('X_train shape:', X_train.shape)
 #  print('Y_train shape:', Y_train.shape)
 #  print('Training on %s images, testing on %s images' % \
 #          (X_train.shape[0], X_test.shape[0]))

 #  model.fit(
 #          X_train, Y_train, batch_size=32,
 #          nb_epoch=nb_epoch, verbose=1)

    return model

main()
