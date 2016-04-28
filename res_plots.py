'''Plots for Yelp Paper'''
import matplotlib.pyplot as plt
import numpy as np

# use this to make plot pretty
plt.style.use('ggplot')

CNN = [0.0, 0.743323908263, 0.78677585684, 0.0, 0.0,
       0.846597462514, 0.871650211566, 0.0, 0.67637326274]

NN = [0.676470588235, 0.830769230769,
      0.90625, 0.702222222222, 0.730434782609,
      0.911392405063, 0.917293233083, 0.72, 0.872427983539]

SVM = [0.681481481481, 0.825870646766,
       0.904255319149, 0.673267326733,
       0.747474747475, 0.884297520661, 0.928838951311,
       0.697247706422, 0.896]

LR = [0.696296296296, 0.83,
      0.904255319149, 0.679802955665,
      0.752475247525, 0.877049180328, 0.917293233083,
      0.691588785047, 0.896]

RAND_FOR = [0.701492537313, 0.821621621622,
            0.891304347826, 0.608695652174,
            0.644444444444, 0.910569105691, 0.90977443609,
            0.68085106383, 0.883018867925]

def individual():
    '''Plots F1 score by class for each model. One plot for each model'''
    n_groups = 9
    y_labels = ('0', '1', '2', '3', '4', '5', '6', '7', '8')
    index = np.arange(n_groups)

    # NN    
    fig, ax = plt.subplots()
    nn_plot = plt.bar(index, NN)

    plt.xlabel('Class')
    plt.ylabel('Scores')
    plt.title('F1 score by class for the Neural Network')
    plt.xticks(index + 0.4, y_labels)

    plt.savefig("./plots/nn.png", dpi=600)

    # SVM
    fig, ax = plt.subplots()

    svm_plot = plt.bar(index, SVM)

    plt.xlabel('Class')
    plt.ylabel('Scores')
    plt.title('F1 score by class for SVMs')
    plt.xticks(index + 0.4, y_labels)

    plt.savefig("./plots/svm.png", dpi=600)

    # LOGISTIC REGRESSION
    fig, ax = plt.subplots()

    lr_plot = plt.bar(index, LR)

    plt.xlabel('Class')
    plt.ylabel('Scores')
    plt.title('F1 score by class for Logistic regression')
    plt.xticks(index + 0.4, y_labels)
    
    plt.savefig("./plots/lr.png", dpi=600)

    # RANDOM FORESTS
    fig, ax = plt.subplots()
    
    rand_for_plot = plt.bar(index, RAND_FOR)
    
    plt.xlabel('Class')
    plt.ylabel('Scores')
    plt.title('F1 score by class for Random Forests')
    plt.xticks(index + 0.4, y_labels)
    
    plt.savefig("./plots/rand_for.png", dpi=600)


def grouped():
    '''Plots 1 plot with F1 score by class for each model.
    Only one plot, used to compare models.'''
    n_groups = 9
    y_labels = ('0', '1', '2', '3', '4', '5', '6', '7', '8')
    index = np.arange(n_groups)

    fig, ax = plt.subplots()
    bar_width = 0.15

    CNN_plot = plt.bar(index, CNN, bar_width, label='CNN', color='#D55E00')
    NN_plot = plt.bar(index + 1*bar_width, NN, bar_width, label='Neural\nNetwork', color='#999999')
    SVM_plot = plt.bar(index + 2*bar_width, SVM, bar_width, label='SVM', color='#E69F00')
    LR_plot = plt.bar(index + 3*bar_width, LR, bar_width, label='Logistic\nRegression', color='#56B4E9')
    RAND_FOR_plot = plt.bar(index + 4*bar_width, RAND_FOR, bar_width, label='Random\nForest', color='#009E73')

    plt.xlabel('Class')
    plt.ylabel('Scores')
    plt.title('F1 score by class for all five models')
    plt.xticks(index + 0.4, y_labels)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("./plots/all.png", dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


def mean_compare():
    '''Plots the F1 score and validation set accuracy
    for each model on one plot'''
    # n_algo = 4
    index = np.arange(2)
    y_labels = ('Validation Set Accuracy', 'F1 score')
    cnn = [0.636888888889, 0.665951139732]
    n_net = [0.8322, 0.8258]
    svm = [0.8372, 0.8269]
    log_reg = [0.8366, 0.8264]
    rand_for = [0.8311, 0.8155]

    fig, ax = plt.subplots()
    ax.set_ylim([0.0, 1.0])
    bar_width = 0.15
    cnn_plot = plt.bar(index, cnn, bar_width, label='CNN', color='#D55E00')
    nn_plot = plt.bar(index + 1*bar_width, n_net, bar_width, label='Neural\nNetwork', color='#999999')
    svm_plot = plt.bar(index + 2*bar_width, svm, bar_width, label='SVM', color='#E69F00')
    lr_plot = plt.bar(index + 3*bar_width, log_reg, bar_width, label='Logistic\nRegression', color='#56B4E9')
    rf_plot = plt.bar(index + 4*bar_width, rand_for, bar_width, label='Random\nForest', color='#009E73')

    plt.xlabel('')
    plt.ylabel('Scores')
    plt.title('Scores by model')
    plt.xticks(index + 0.4, y_labels)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("./plots/acc_and_f1.png", dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


def main():
    individual()
    grouped()
    mean_compare()

if __name__ == '__main__':
    main()   
