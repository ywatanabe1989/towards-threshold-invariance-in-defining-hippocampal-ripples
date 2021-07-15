#!/usr/bin/env python

from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import utils


class LearningCurveLogger(object):
    def __init__(
        self,
    ):
        self.steps_str = ("Training", "Test")
        self.logged_dict = utils.general.listed_dict(self.steps_str)
        for k in self.steps_str:
            self.logged_dict[k] = {}

    def __call__(self, log_dict, step="Training"):
        for k in log_dict.keys():
            try:
                self.logged_dict[step][k].append(log_dict[k])
            except:
                self.logged_dict[step][k] = []
                self.logged_dict[step][k].append(log_dict[k])

    def to_dfs(self, steps_str=["Training", "Test"]):  # fixme
        self.dfs = {}
        self.dfs_pivot = {}
        for step in steps_str:
            logged_dict = self.logged_dict[step]
            length = len(logged_dict["i_epoch"])
            df = pd.DataFrame(
                {
                    "step": [step for _ in range(length)],
                    "i_mouse_test": logged_dict["i_mouse_test"],
                    "i_epoch": logged_dict["i_epoch"],
                    "i_global": logged_dict["i_global"],
                    "loss": logged_dict["loss"],
                    "ACC": logged_dict["ACC"],
                    "pred_proba": [logged_dict["pred_proba"][i] for i in range(length)],
                    "gt_label": [logged_dict["gt_label"][i] for i in range(length)],
                }
            )

            self.dfs[step] = df

            # pivot version to plot
            self.dfs_pivot[step] = (
                self.dfs[step][
                    [
                        "i_global",
                        "ACC",
                        "loss",
                    ]
                ]
                .pivot_table(columns=["i_global"], aggfunc="mean")
                .T
            )

    def print_learning_curves_in_digits(self, step="Training"):
        print(step)

        try:
            df = pd.DataFrame(
                {
                    "i_epoch": self.logged_dict[step]["i_epoch"],
                    "ACC": self.logged_dict[step]["ACC"],
                }
            ).pivot_table(columns=["i_epoch"], aggfunc="mean")
            pprint(df)
        except:
            pass

        print()

    def plot_learning_curves(
        self,
        i_mouse_test=None,
        max_epochs=None,
        window_size_sec=None,
        figscale=10,
    ):

        self.to_dfs()

        ## Plot
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)
        title = "fold#{}; MAX EPOCHS = {}; WINDOW SIZE = {} [sec]".format(
            i_mouse_test, max_epochs, window_size_sec
        )
        fig.text(0.5, 0.95, title, ha="center")

        keys_to_plot = ["loss", "ACC"]

        COLOR_DICT = {
            "Training": "blue",
            "Test": "green",
        }

        for i_plt, plt_k in enumerate(keys_to_plot):
            for step_k in self.dfs_pivot.keys():
                if step_k == "Training":  # line
                    ax[i_plt].plot(
                        self.dfs_pivot[step_k].index,
                        self.dfs_pivot[step_k][plt_k],
                        label=step_k,
                        color=COLOR_DICT[step_k],
                    )

                if step_k == "Test":  # scatter
                    ax[i_plt].scatter(
                        self.dfs_pivot[step_k].index,
                        self.dfs_pivot[step_k][plt_k],
                        label=step_k,
                        color=COLOR_DICT[step_k],
                        s=150,
                        alpha=0.9,
                    )

                if "ACC" in plt_k:
                    ax[i_plt].set_ylim(0, 1)

        ax[0].legend()
        ax[1].legend()

        ax[0].set_xlabel("Iteration#")

        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Accuracy")

        return fig


if __name__ == "__main__":
    ################################################################################
    ## MNIST
    ################################################################################
    from sklearn import datasets, metrics, svm
    from sklearn.model_selection import train_test_split

    digits = datasets.load_digits()

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    MODEL_CONF = utils.general.load("models/ResNet1D/ResNet1D.yaml")
    MODEL_CONF["SEQ_LEN"] = data.shape[-1]
    MODEL_CONF["LABELS"] = [i for i in range(len(np.unique(digits.target)))]
    model = ResNet1D(MODEL_CONF)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    # # Learn the digits on the train subset
    # clf.fit(X_train, y_train)

    import torch
    from scipy.special import softmax

    lc_logger = LearningCurveLogger()
    i_global = 0
    batch_size = 8
    i_mouse_test = 0
    max_epochs = 10

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for i_epoch in range(max_epochs):
        step = "Training"
        model.train()
        for i_batch in range(100):
            start = i_batch * batch_size
            end = (i_batch + 1) * batch_size
            Xb = X_train[start:end]
            Tb = y_train[start:end]

            logits = model(torch.tensor(Xb).float())
            pred_proba = softmax(logits.detach().numpy(), axis=-1)
            pred_class = pred_proba.argmax(axis=-1)

            loss = loss_fn(logits, torch.tensor(Tb).long())

            loss.backward()
            optimizer.step()

            acc = (pred_class == Tb).mean()

            log_dict = {
                "loss": float(loss.detach().numpy()),
                "ACC": float(acc),
                "pred_proba": pred_proba,
                "gt_label": Tb,
                "i_mouse_test": i_mouse_test,
                "i_epoch": i_epoch,
                "i_global": i_global,
            }

            lc_logger(log_dict, step=step)

            i_global += 1

    step = "Test"
    model.eval()
    for i_batch in range(100):
        start = i_batch * batch_size
        end = (i_batch + 1) * batch_size
        Xb = X_test[start:end]
        Tb = y_test[start:end]

        logits = model(torch.tensor(Xb).float())
        pred_proba = softmax(logits.detach().numpy(), axis=-1)
        pred_class = pred_proba.argmax(axis=-1)

        loss = loss_fn(logits, torch.tensor(Tb).long())

        # loss.backward()
        # optimizer.step()

        acc = (pred_class == Tb).mean()

        log_dict = {
            "loss": float(loss.detach().numpy()),
            "ACC": float(acc),
            "pred_proba": pred_proba,
            "gt_label": Tb,
            "i_mouse_test": i_mouse_test,
            "i_epoch": i_epoch,
            "i_global": i_global,
        }

        lc_logger(log_dict, step=step)

    fig = lc_logger.plot_learning_curves(
        i_mouse_test=i_mouse_test,
        max_epochs=max_epochs,
        window_size_sec=1024,
        figscale=10,
    )
    fig.show()
