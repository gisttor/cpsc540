from __future__ import print_function
import numpy as np
import sys
np.random.seed(1337)  # for reproducibility

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import cross_validation
import json
import score
import read_data
import math

def main():
    x_train, y_train, x_test, y_test = load_features()
    x_train_conc = concate_x(x_train)
    x_test_conc = concate_x(x_test)

    # log_reg = LogisticRegressionCV(verbose=1, penalty='l2', n_jobs=1)
    # linear_svc = linear_svc_cv(x_train_conc, y_train)
    rand_for = random_forest_cv(x_train_conc, y_train)

    # y_hat = one_vs_rest(x_train_conc, y_train, x_test_conc, log_reg)

    y_hat = one_vs_rest(x_train_conc, y_train, x_test_conc, rand_for)
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

def linear_svc_cv(x_train, y_train):
    '''Returns a Linear SVC with the regularization parameter (C)
    selected by cross validation'''
    lambdas = [math.pow(2, x) for x in range(-4, 6)]
    cv_scores = []
    for lam in lambdas:
        print('lambda: ', lam)
        linear_svc = OneVsRestClassifier(LinearSVC(verbose=0, C=lam))
        scores = cross_validation.cross_val_score(
            linear_svc, x_train, y_train,
            cv=3, scoring='average_precision')
        print(scores.mean())
        cv_scores.append(scores.mean())

    print(cv_scores)

    arg_max = np.argmax(cv_scores)
    final_lambda = lambdas[arg_max]
    return LinearSVC(verbose=1, C=final_lambda)

def random_forest_cv(x_train, y_train):
    '''Returns a Random Forest Classifier with n_estimators
    selected by cross validation'''
    n_trees = [200 + 45*x for x in range(0, 6)]
    cv_scores = []
    for n_tree in n_trees:
        print('n_trees: ', n_tree)
        rand_for = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=n_tree,
                max_features='auto',
                max_depth=None,
                n_jobs=-1,
                verbose=0,
                criterion="entropy"))
        scores = cross_validation.cross_val_score(
            rand_for, x_train, y_train,
            cv=3,
            scoring='average_precision')
        print(scores.mean())
        cv_scores.append(scores.mean())

    print(cv_scores)

    arg_max = np.argmax(cv_scores)
    final_n_trees = n_trees[arg_max]
    return RandomForestClassifier(
        n_estimators=final_n_trees,
        max_features='auto',
        max_depth=None,
        n_jobs=-1,
        verbose=1,
        criterion="entropy")

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
