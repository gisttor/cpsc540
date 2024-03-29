import numpy as np

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

def f1_by_class(y_pred, y_test):
    score = []
    for i in range(y_pred.shape[1]):
        score.append(f1score(y_pred[:,i], y_test[:,i]))
    return score
