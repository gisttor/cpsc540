from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import read_data

batch_size = 128
nb_classes = 9
nb_epoch = 12

# input image dimensions
img_size = 100
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 10

# the data, shuffled and split between train and test sets
X_train, Y_train = read_data.read_data_photo_labels(img_size = img_size, num_biz = 100)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
			input_shape=(3, img_size, img_size)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('tanh'))

model.compile(loss='mae', optimizer='adagrad')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
	  show_accuracy=True, verbose=1, validation_split = 0.1)
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
