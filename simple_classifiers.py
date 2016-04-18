from __future__ import print_function
import numpy as np
import sys
np.random.seed(1337)  # for reproducibility

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegressionCV
import json
import score
import read_data

def main():
    x_train, y_train, x_test, y_test = load_features()
    x_train_conc = concate_x(x_train)
    x_test_conc = concate_x(x_test)

    log_reg = LogisticRegressionCV(verbose=1, penalty='l2', n_jobs=1)
    rand_for = RandomForestClassifier(
        n_estimators=300,
        max_features='auto',
        max_depth=None,
        n_jobs=-1,
        verbose=1,
        criterion="entropy")
    linear_svc = LinearSVC(verbose=1)
    
    y_hat = one_vs_rest(x_train_conc, y_train, x_test_conc, log_reg)

    # y_hat = one_vs_rest(x_train_conc, y_train, x_test_conc, rand_for)
    # y_hat = one_vs_rest(x_train_conc, y_train, x_test_conc, linear_svc)
    # y_hat = one_for_each_class(x_train_conc, y_train, x_test_conc, rand_for)
    # y_hat = one_for_each_class(x_train_conc, y_train, x_test_conc, linear_svc)

    # np.savetxt('test.csv', y_test, delimiter=',')
    # np.savetxt('hat.csv', y_hat, delimiter=',')

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

def one_for_each_class(x_train, y_train, x_test, clf):
    '''Fit one clf for each class label and predict independantly'''
    y_hat = []
    num_classes = len(y_train[1, :])
    for i in range(num_classes):
        print("creating classifier:", i)
        print("fitting classifier:", i)
        clf.fit(x_train, y_train[:,i])
        print("getting predictions for attribute:", i)
        y_hat.append(clf.predict(x_test))
    
    y_hat = np.vstack(y_hat)
    y_hat = np.transpose(y_hat)
    return y_hat

def one_vs_rest(x_train, y_train, x_test, clf):
    '''Fit one vs rest classifier and predict'''
    clf = OneVsRestClassifier(clf)
    print("fitting classifier")
    clf.fit(x_train, y_train)
    print("getting predictions")
    y_hat = clf.predict(x_test)
    y_hat[y_hat == 0] = -1
    return y_hat

def concate_x(X):
    # just combine all the Xs (GoogLeNet uses 3 classifiers)
    X_conc = np.concatenate((X[:,0,:], X[:,1,:]), axis=1)
    return np.concatenate((X_conc, X[:,2,:]), axis=1)

if __name__ == '__main__':
    main()
