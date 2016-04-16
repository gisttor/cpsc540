from __future__ import print_function
import numpy as np
import sys
np.random.seed(1337)  # for reproducibility

from sklearn.ensemble import RandomForestClassifier
import json
import score
import read_data

def main():
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


    # this is where the random forest part starts
    # in this program we train a random forest for each label
    # and predict independantly

    y_pred = []
    num_classes = len(y_train[1, :])
    for i in range(num_classes):
        print("creating classifier:", i)
        # just combine all the x_trains (GoogLeNet uses 3 classifiers)
        x_train_conc = np.concatenate((x_train[:,0,:], x_train[:,1,:]), axis=1)
        x_train_conc = np.concatenate((x_train_conc, x_train[:,2,:]), axis=1)
        
        random_forest = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            oob_score=True,
            verbose=1,
            criterion="entropy")

        print("fitting classifier:", i)
        random_forest.fit(x_train_conc, y_train[:,i])

        print("getting predictions for attribute:", i)
        x_test_conc = np.concatenate((x_test[:,0,:], x_test[:,1,:]), axis=1)
        x_test_conc = np.concatenate((x_test_conc, x_test[:,2,:]), axis=1)
        y_pred.append(random_forest.predict(x_test_conc))

    print("preparing output...")
    y_pred = np.vstack(y_pred)
    y_pred = np.transpose(y_pred)

    # np.savetxt('pred.csv', y_pred, delimiter=',')

    print('Test accuracy: ', score.accuracy(y_pred, y_test))
    print('F1 score: ', score.f1score(y_pred, y_test))
    print('F1 score by class:')
    score_byclass = score.f1_by_class(y_pred, y_test)
    for c, class_score in enumerate(score_byclass):
        print(c, ':', class_score)


if __name__ == '__main__':
    main()
