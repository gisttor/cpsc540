from __future__ import print_function
import numpy as np
import sys
np.random.seed(1337)  # for reproducibility

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import json
import score
import read_data

def main():
    x_train, y_train, x_test, y_test = load_features()
    x_train_conc = concate_x(x_train)
    x_test_conc = concate_x(x_test)

    y_hat = svm(x_train_conc, y_train, x_test_conc)
    # y_hat = random_forests(x_train_conc, y_train, x_test_conc)

    # np.savetxt('pred.csv', y_hat, delimiter=',')
    print('Test accuracy: ', score.accuracy(y_hat, y_test))
    print('F1 score: ', score.f1score(y_hat, y_test))
    print('F1 score by class:')
    score_byclass = score.f1_by_class(y_hat, y_test)
    for c, class_score in enumerate(score_byclass):
        print(c, ':', class_score)

def load_features():
    # this is just copy pasted from net2.py

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

    return x_train, y_train, x_test, y_test

def random_forests(x_train, y_train, x_test):
    '''Fit random forests to each label class and predict independantly'''
    y_hat = []
    num_classes = len(y_train[1, :])
    for i in range(num_classes):
        print("creating classifier:", i)
        random_forest = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            oob_score=True,
            verbose=1,
            criterion="entropy")
        print("fitting classifier:", i)
        random_forest.fit(x_train, y_train[:,i])
        print("getting predictions for attribute:", i)
        y_hat.append(random_forest.predict(x_test))

    print("preparing output...")
    y_hat = np.vstack(y_hat)
    y_hat = np.transpose(y_hat)
    return y_hat

def svm(x_train, y_train, x_test):
    '''Fit linear SVMs to each label class and predict independantly'''
    y_hat = []
    num_classes = len(y_train[1, :])
    for i in range(num_classes):
        print("creating classifier:", i)
        clf = LinearSVC(verbose=1)
        print("fitting classifier:", i)
        clf.fit(x_train, y_train[:,i])
        print("getting predictions for attribute:", i)
        y_hat.append(clf.predict(x_test))
    
    y_hat = np.vstack(y_hat)
    y_hat = np.transpose(y_hat)
    return y_hat

def concate_x(X):
    # just combine all the Xs (GoogLeNet uses 3 classifiers)
    X_conc = np.concatenate((X[:,0,:], X[:,1,:]), axis=1)
    return np.concatenate((X_conc, X[:,2,:]), axis=1)

if __name__ == '__main__':
    main()
