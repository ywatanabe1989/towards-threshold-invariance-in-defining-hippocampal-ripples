#!/usr/bin/env python

#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# def plot_cm(cm, labels=None, spath=None):
#     ax = plt.subplot()
#     sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')
#     ax.set_xlabel('Predicted labels')
#     ax.set_ylabel('True labels')
#     ax.set_title('Confusion Matrix')
#     if labels is not None:
#         ax.xaxis.set_ticklabels(labels)
#         ax.yaxis.set_ticklabels(labels)

#     if spath is not None:
#         plt.savefig(spath, dpi=300)
#         print('Saved to: {}'.format(spath))
#     else:
#         plt.show()


def confusion_matrix(cm, labels=None, title=None, show=False):
    """
    Inverse the y-axis and plot the confusion matrix as a heatmap.
    """

    if title is None:
        title = "Confusion Matrix"

    df = pd.DataFrame(data=cm)

    if labels is not None:
        df.index = labels
        df.columns = labels

    fig, ax = plt.subplots()
    res = sns.heatmap(
        df,
        annot=True,
        fmt=".0f",
        cmap="Blues",
    )  # cbar_kws={"shrink": 0.82}
    res.invert_yaxis()

    # make frame visible
    for _, spine in res.spines.items():
        spine.set_visible(True)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    if show:
        fig.show()

    return fig

    # cm = cm[::-1]
    # if title is None:
    #     title = "Confusion Matrix"

    # fig, ax = plt.subplots()
    # sns.heatmap(cm, annot=True, ax=ax, cmap="Blues")
    # ax.set_xlabel("Predicted label")
    # ax.set_ylabel("True label")
    # ax.set_title(title)

    # if labels is not None:
    #     ax.xaxis.set_ticklabels(labels)
    #     ax.yaxis.set_ticklabels(labels)

    # if spath is not None:
    #     plt.savefig(spath, dpi=300)
    #     print("Saved to: {}".format(spath))
    # else:
    #     plt.show()


if __name__ == "__main__":
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn
    from sklearn import datasets, svm
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.model_selection import train_test_split

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel="linear", C=0.01).fit(X_train, y_train)

    ## unique
    y_pred = classifier.predict(X_test)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    confusion_matrix(cm, labels=class_names, show=True)

    ## EOF
