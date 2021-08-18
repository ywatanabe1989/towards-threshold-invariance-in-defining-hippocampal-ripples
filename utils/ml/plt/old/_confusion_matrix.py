#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker


def confusion_matrix(cm, labels=None, title=None, colorbar=True, extend_ratio=1.0):
    """
    Inverse the y-axis and plot the confusion matrix as a heatmap.

    kwargs:

        "extend_ratio":
            determines how much the axes objects (not the fig object) are expanded
            both horizontally and vertically.
    """

    df = pd.DataFrame(data=cm)
    vmax = int(max(df))

    # x- and y-ticklabels
    if labels is not None:
        df.index = df.columns = labels

    fig, ax = plt.subplots()
    res = sns.heatmap(
        df,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=False,
    )  # Here, don't plot color bar.

    # Inverts the y-axis
    res.invert_yaxis()

    # Makes the frame visible
    for _, spine in res.spines.items():
        spine.set_visible(True)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    ax.set_box_aspect(1)

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=60,
        fontdict={"horizontalalignment": "right"},
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=60,
        fontdict={"verticalalignment": "top"},
    )

    # The size of the confusion matrix
    ax = utils.plt.ax_extend(ax, extend_ratio, extend_ratio)

    """
    The axes objects of the confusion matrix and colorbar are different.
    Here, their sizes are adjusted one by one.
    """
    if colorbar == True:
        # Plots colorbar and adjusts the size
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="Blues"), ax=ax, shrink=0.68
        )
        cb.locator = ticker.MaxNLocator(nbins=4)  # tick_locator
        cb.update_ticks()
        # fig.axes[-1] = utils.plt.ax_extend(fig.axes[-1], 1, extend_ratio)

    return fig


if __name__ == "__main__":
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn
    from sklearn import datasets, svm
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.model_selection import train_test_split

    sys.path.append(".")
    import utils

    utils.plt.configure_mpl(
        plt,
        dpi=100,
        figsize=(4, 8),
        fontsize=7,
        legendfontsize="xx-small",
        hide_spines=True,
        tick_size=0.8,
        tick_width=0.2,
    )

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
    fig = confusion_matrix(cm, labels=class_names, extend_ratio=0.2, colorbar=True)

    fig.show()

    ## EOF
