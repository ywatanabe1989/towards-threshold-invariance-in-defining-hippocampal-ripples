#!/usr/bin/env python
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import utils
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


def confusion_matrix(
    plt,
    cm,
    labels=None,
    label_rotation_xy=(0, 0),
    title=None,
    colorbar=True,
    extend_ratio=1.0,
):
    """
    Inverse the y-axis and plot the confusion matrix as a heatmap.
    The predicted labels (in x-axis) is symboled with hat (^).
    The plt object is passed to adjust the figure size

    kwargs:

        "extend_ratio":
            Determines how much the axes objects (not the fig object) are extended
            in the vertical direction.

    """
    df = pd.DataFrame(data=cm)
    vmax = np.array(df).max().astype(int)

    # x- and y-ticklabels
    if labels is not None:
        df.index = [utils.general.to_the_latex_style(l) for l in labels]  # true labels
        df.columns = [
            utils.general.add_hat_in_the_latex_style(l) for l in labels
        ]  # predicted labels

    fig, ax = plt.subplots()
    res = sns.heatmap(
        df,
        annot=True,
        annot_kws={"size": plt.rcParams["font.size"]},
        fmt=".0f",
        cmap="Blues",
        cbar=False,
    )  # Here, don't plot color bar.

    ## Adds comma separator for the annotated int texts
    for t in ax.texts:
        t.set_text("{:,d}".format(int(t.get_text())))

    # Inverts the y-axis
    res.invert_yaxis()

    # Makes the frame visible
    for _, spine in res.spines.items():
        spine.set_visible(True)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    ax = utils.plt.ax_extend(ax, 1, extend_ratio)

    ax.set_box_aspect(1)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=label_rotation_xy[0],
        fontdict={"verticalalignment": "top"},
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=label_rotation_xy[1],
        fontdict={"horizontalalignment": "right"},
    )

    # The size of the confusion matrix

    # Calculates the dx
    bbox = ax.get_position()
    left_orig = bbox.x0
    width_orig = bbox.x1 - bbox.x0
    g_x_orig = left_orig + width_orig / 2.0
    width_tgt = width_orig * extend_ratio  # x_extend_ratio
    dx = width_orig - width_tgt
    # print(dx)

    """
    The axes objects of the confusion matrix and colorbar are different.
    Here, their sizes are adjusted one by one.
    """
    if colorbar == True:  # fixme
        divider = make_axes_locatable(ax)  # Gets region from the ax
        cax = divider.append_axes("right", size="5%", pad=0.1)
        # cax = divider.new_horizontal(size="5%", pad=1, pack_start=True)
        cax = utils.plt.ax_set_position(fig, cax, -dx * 2.54, 0)
        fig.add_axes(cax)

        """
        axpos = ax.get_position()        
        caxpos = cax.get_position()

        AddAxesBBoxRect(fig, ax, ec="r")
        AddAxesBBoxRect(fig, cax, ec="b")

        fig.text(
            axpos.x0 + 0.01, axpos.y0 + 0.01, "after colorbar", weight="bold", color="r"
        )

        fig.text(
            caxpos.x1 + 0.01,
            caxpos.y1 - 0.01,
            "cax position",
            va="top",
            weight="bold",
            color="b",
            rotation="vertical",
        )
        """

        # Plots colorbar and adjusts the size
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="Blues"),
            cax=cax,
            # shrink=0.68,
        )
        cbar.locator = ticker.MaxNLocator(nbins=4)  # tick_locator
        cbar.update_ticks()

    return fig


# def AddAxesBBoxRect(fig, ax, ec="k"):
#     from matplotlib.patches import Rectangle

#     axpos = ax.get_position()
#     rect = fig.patches.append(
#         Rectangle(
#             (axpos.x0, axpos.y0),
#             axpos.width,
#             axpos.height,
#             ls="solid",
#             lw=2,
#             ec=ec,
#             fill=False,
#             transform=fig.transFigure,
#         )
#     )
#     return rect


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

    # Imports some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Splits the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Runs classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel="linear", C=0.01).fit(X_train, y_train)

    ## Checks the confusion_matrix function
    y_pred = classifier.predict(X_test)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    cm **= 3

    utils.plt.configure_mpl(
        plt,
        # figsize=(4, 8),
        figsize=(4, 8),
        fontsize=6,
        labelsize=8,
        legendfontsize=7,
        tick_size=0.8,
        tick_width=0.2,
    )

    labels = class_names

    fig = confusion_matrix(
        plt,
        cm,
        labels=class_names,
        label_rotation_xy=(60, 60),
        extend_ratio=0.5,
        colorbar=True,
    )

    fig.axes[-1] = utils.plt.ax_scientific_notation(
        fig.axes[-1],
        3,
        fformat="%3.1f",
        # fformat="%d",
        y=True,
    )

    fig.show()

    ## EOF
