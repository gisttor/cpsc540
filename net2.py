from __future__ import print_function
import numpy as np
import sys
np.random.seed(1337)  # for reproducibility

from keras import models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import json
import score
import read_data

if __name__ == '__main__':
    dropout = 0.5
    batch_size = 64
    nb_classes = 9
    nb_epoch = 40

    model = Sequential()

    model.add(Flatten(input_shape=(3,1000)))

    model.add(Dropout(dropout))
    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dropout(dropout))
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(nb_classes))

    model.compile(loss='hinge', optimizer='adam')

    # image_labels: shape (num_image, 3, 1000)
    image_labels = np.load('data/googlenet_predictions.npy')
    with open('data/googlenet_predictions_order.json', 'r') as jfile:
        image_label_order = json.load(jfile)
    biz_csv = read_data.read_biz_csv()

    num_biz = len(biz_csv)
    x_whole = np.zeros((num_biz, image_labels.shape[1], image_labels.shape[2])) 
    y_whole = np.zeros((num_biz, 9))
    # aggregate data by business: take average of predictions
    for idx, (biz_id, (label, photo_ids)) in enumerate(biz_csv.items()):
        for photo_id in photo_ids:
            image_idx = image_label_order[str(photo_id)]
            x_whole[idx] += image_labels[image_idx]
        x_whole[idx] /= len(photo_ids)
        y_whole[idx] = label

    train_frac = 0.9
    x_train, x_test = np.vsplit(x_whole, [int(num_biz*train_frac)])
    y_train, y_test = np.vsplit(y_whole, [int(num_biz*train_frac)])

    print('X_train shape:', x_train.shape)
    print('Y_train shape:', y_train.shape)
    print('Training on %s biz, testing on %s biz' % \
            (x_train.shape[0], x_test.shape[0]))

    model.fit(x_train, y_train, batch_size=batch_size,
            nb_epoch=nb_epoch, verbose=1, validation_split=0.0)

    test_pred = np.sign(model.predict(x_test))
    test_loss = model.evaluate(x_test, y_test)
    np.savetxt('pred.csv', test_pred, delimiter=',')

    print('Test loss: ', test_loss)
    print('Test accuracy: ', score.accuracy(test_pred, y_test))
    print('F1 score: ', score.f1score(test_pred, y_test))
    print('F1 score by class:')
    score_byclass = score.f1_by_class(test_pred, y_test)
    for c, score in enumerate(score_byclass):
        print(c, ':', score)
