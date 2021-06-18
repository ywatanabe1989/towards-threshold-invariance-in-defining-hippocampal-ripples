#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (average_precision_score,
                             precision_recall_curve,
                             )
from itertools import cycle



def calc_pr_auc(true_class, pred_proba, labels, plot=False):

    ## One-hot encoding    
    def to_onehot(labels, n_classes):
        eye = np.eye(n_classes, dtype=int)
        return eye[labels]

    # Use label_binarize to be multi-label like settings
    n_classes = len(labels)
    true_class = to_onehot(true_class, n_classes)

    # For each class
    precision = dict()
    recall = dict()
    threshold = dict()
    pr_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(true_class[:, i],
                                                                       pred_proba[:, i])
        pr_auc[i] = average_precision_score(true_class[:, i], pred_proba[:, i])


    ################################################################################
    ## Average precision: micro and macro
    ################################################################################    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], threshold["micro"] = \
        precision_recall_curve(true_class.ravel(), pred_proba.ravel())
    pr_auc["micro"] = \
        average_precision_score(true_class, pred_proba, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(pr_auc["micro"]))

    # macro
    pr_auc["macro"] = \
        average_precision_score(true_class, pred_proba, average="macro")
    print('Average precision score, macro-averaged over all classes: {0:0.2f}'
          .format(pr_auc["macro"]))

    if plot:
        # ## Configures matplotlib
        # plt.rcParams['font.size'] = 20
        # plt.rcParams['figure.figsize'] = (16*1.2, 9*1.2)
        
        # Plot Precision-Recall curve for each class and iso-f1 curves
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        fig, ax = plt.subplots()
        ax.set_box_aspect(1)
        lines = []
        legends = []

        # iso-F1: By definition, an iso-F1 curve contains all points
        #         in the precision/recall space whose F1 scores are the same.
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
            # plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))            

        lines.append(l)
        legends.append('iso-f1 curves')
        l, = ax.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        legends.append('micro-average (area = {0:0.2f})'
                      ''.format(pr_auc["micro"]))

        ## Each Class
        for i, color in zip(range(n_classes), colors):
            l, = ax.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            legends.append('{0} (area = {1:0.2f})'
                          ''.format(labels[i], pr_auc[i]))
         
        # fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(lines, legends, loc='lower left')
         
        fig.show()

    return pr_auc, precision, recall, threshold        


if __name__ == '__main__':
    import numpy as np
    from scipy.special import softmax    
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split

    def mk_demo_data(n_classes=2, batch_size=16):
        labels = ['cls{}'.format(i_cls) for i_cls in range(n_classes)]
        true_class = np.random.randint(0, n_classes, size=(batch_size,))
        pred_proba = softmax(np.random.rand(batch_size, n_classes), axis=-1) 
        pred_class = np.argmax(pred_proba, axis=-1)
        return labels, true_class, pred_proba, pred_class

    ## Fix seed
    np.random.seed(42)


    '''
    ################################################################################
    ## A Minimal Example
    ################################################################################    
    labels, true_class, pred_proba, pred_class = \
        mk_demo_data(n_classes=10, batch_size=256)

    pr_auc, precision, recall, threshold = \
        calc_pr_auc(true_class, pred_proba, labels, plot=False)
    '''

    ################################################################################
    ## MNIST
    ################################################################################
    from sklearn import datasets, svm, metrics
    from sklearn.model_selection import train_test_split
    digits = datasets.load_digits()

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001, probability=True)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted_proba = clf.predict_proba(X_test)    
    predicted = clf.predict(X_test)
    
    n_classes = len(np.unique(digits.target))
    labels = ['Class {}'.format(i) for i in range(n_classes)]

    ## Configures matplotlib
    plt.rcParams['font.size'] = 20
    plt.rcParams['legend.fontsize'] = 'xx-small'
    plt.rcParams['figure.figsize'] = (16*1.2, 9*1.2)
    
    ## Main
    pr_auc, precision, recall, threshold = \
        calc_pr_auc(y_test, predicted_proba, labels, plot=True)
