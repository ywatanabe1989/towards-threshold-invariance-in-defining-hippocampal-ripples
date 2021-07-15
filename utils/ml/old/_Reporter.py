#!/usr/bin/env python

import os

import numpy as np
import pandas as pd
# from utils.general import (mv_to_tmp,
#                            torch_to_arr,
#                            save_listed_scalars_as_csv,
#                            save_listed_dfs_as_csv,
#                            TimeStamper,
#                            )
# import utils.general as ug
# from utils.ml import plot_cm
import utils.ml as um
import yaml
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score)


class Reporter:
    """Saved Confusion Matrix, Classification Report, and ROC-AUC score under self.sdir."""

    def __init__(self, sdir):
        ## Make save dir
        self.ts = utils.general.TimeStamper()
        self.sdir = sdir
        self.conf_mats_folds = []
        self.clf_reports_folds = []
        self.pr_aucs_folds = []
        self.roc_aucs_folds = []

        self.ts("Reporter has been initialized.")

    def calc_metrics(
        self, true_class, pred_class, pred_proba, labels=None, i_fold=None
    ):
        """Calculates ACC, Confusion Matrix, Classification Report, and ROC-AUC score."""
        print(
            "\n ---------------------------------------------------------------------- \n"
        )
        self.labels = labels

        true_class, pred_class, pred_proba = (
            utils.general.torch_to_arr(true_class),
            utils.general.torch_to_arr(pred_class),
            utils.general.torch_to_arr(pred_proba),
        )

        ##############################
        ## ACC ##
        ##############################
        # import pdb; pdb.set_trace()
        # from sklearn.metrics import accuracy_score
        # acc = accuracy_score(true_class.reshape(-1), pred_class.reshape(-1))
        acc = (true_class.reshape(-1) == pred_class.reshape(-1)).mean()
        print("\nACC in fold#{} was {:.3f}\n".format(i_fold, acc))

        ##############################
        ## Confusion Matrix ##
        ##############################
        conf_mat = confusion_matrix(true_class, pred_class)
        conf_mat = pd.DataFrame(data=conf_mat, columns=labels).set_index(
            pd.Series(list(labels))
        )
        print("\nConfusion Matrix in fold#{}: \n{}\n".format(i_fold, conf_mat))

        ##############################
        ## Classification Report ##
        ##############################
        clf_report = pd.DataFrame(
            classification_report(
                true_class,
                pred_class,
                target_names=labels,
                output_dict=True,
            )
        ).round(3)
        # rename 'support' to 'sample size'
        clf_report["index"] = clf_report.index
        clf_report.loc["support", "index"] = "sample size"
        clf_report.set_index("index", drop=True, inplace=True)
        clf_report.index.name = None
        print("\nClassification Report in fold#{}: \n{}\n".format(i_fold, clf_report))

        ##############################
        ## PRE-REC-AUC score ##
        ##############################
        n_classes = len(labels)
        if n_classes == 2:
            pre, rec, thres = precision_recall_curve(true_class, pred_proba[:, 1])
            pr_auc = auc(rec, pre)  # auc(pre, rec)
        else:
            pr_auc = None
        print("\nPR_AUC in fold#{} was {:.3f}\n".format(i_fold, pr_auc))

        ##############################
        ## ROC-AUC score ##
        ##############################
        n_classes = len(labels)
        if n_classes == 2:
            multi_class = "raise"
            pred_proba = pred_proba[:, 1]
        else:
            multi_class = "ovo"
        roc_auc = roc_auc_score(
            true_class,
            pred_proba,
            multi_class=multi_class,
            average="macro",
        )  # fixme; multi_class for 2-class classification problem
        """
        'ovo' stands for One-vs-one. This option makes it compute the average AUC of
        all possible pairwise combinations of classes. Insensitive to class imbalance
        when average == 'macro'.
        """
        print("\nROC_AUC in fold#{} was {:.3f}\n".format(i_fold, roc_auc))

        ## To attributes
        self.conf_mats_folds.append(conf_mat)
        self.clf_reports_folds.append(clf_report)
        self.pr_aucs_folds.append(pr_auc)
        self.roc_aucs_folds.append(roc_auc)

        self.ts("\ni_fold={} ends.\n".format(i_fold))

        print(
            "\n ---------------------------------------------------------------------- \n"
        )

    def summarize(
        self,
    ):
        ## Summarize each fold's metirics
        self.conf_mat_cv_sum = summarize_dfs(self.conf_mats_folds, method="sum")

        self.clf_report_cv_mean, self.clf_report_cv_std = summarize_dfs(
            self.clf_reports_folds, method="mean"
        )

        self.pr_auc_cv_mean, self.pr_auc_cv_std = take_mean_and_std(self.pr_aucs_folds)

        self.roc_auc_cv_mean, self.roc_auc_cv_std = take_mean_and_std(
            self.roc_aucs_folds
        )

        self.num_folds = len(self.conf_mats_folds)

        print("\n --- {}-fold CV overall metrics --- \n".format(self.num_folds))
        print(
            "\nConfusion Matrix (Test; sum; num. folds={})\n{}\n".format(
                self.num_folds, self.conf_mat_cv_sum
            )
        )
        print(
            "\nClassification Report (Test; mean; num. folds={})\n{}\n".format(
                self.num_folds, self.clf_report_cv_mean
            )
        )
        print(
            "\nClassification Report (Test; std; num. folds={})\n{}\n".format(
                self.num_folds, self.clf_report_cv_std
            )
        )
        print(
            "\nPRE-REC AUC Score: {} +/- {} (mean +/- std.; n={})\n".format(
                self.pr_auc_cv_mean, self.pr_auc_cv_std, self.num_folds
            )
        )
        print(
            "\nROC AUC Score: {} +/- {} (mean +/- std.; n={})\n".format(
                self.roc_auc_cv_mean, self.roc_auc_cv_std, self.num_folds
            )
        )

    def save(self, others_dict=None, meta_dict=None, labels=None):
        os.makedirs(self.sdir, exist_ok=True)

        for k, v in others_dict.items():
            spath = self.sdir + k
            utils.general.save(v, spath)

        if meta_dict is not None:
            spath_meta_yaml = self.sdir + "meta.yaml"
            with open(spath_meta_yaml, "w") as f:
                yaml.dump(meta_dict, f)
            print("Saved to: {}".format(spath_meta_yaml))

        ##############################
        ## Confusion Matrix
        ##############################
        ##########
        # Values
        ##########
        conf_mats_cat = [self.conf_mat_cv_sum] + self.conf_mats_folds
        indi_suffix_cat = ["{}-fold CV SUM".format(self.num_folds)] + [
            "fold#{}".format(i) for i in range(self.num_folds)
        ]
        utils.general.save_listed_dfs_as_csv(
            conf_mats_cat,
            self.sdir + "conf_mats.csv",
            indi_suffix=indi_suffix_cat,
            overwrite=True,
        )
        ##########
        # Figures; fixme; each fold
        ##########
        spath = self.sdir + "conf_mat_overall_sum.png"
        um.plot_cm(self.conf_mat_cv_sum, labels=labels, spath=spath)

        ##############################
        ## Classification Matrix
        ##############################
        clf_reports_cat = (
            [self.clf_report_cv_mean]
            + [self.clf_report_cv_std]
            + self.clf_reports_folds
        )
        indi_suffix_cat = (
            ["{}-fold CV mean".format(self.num_folds)]
            + ["{}-fold CV std.".format(self.num_folds)]
            + ["fold#{}".format(i) for i in range(self.num_folds)]
        )
        utils.general.save_listed_dfs_as_csv(
            clf_reports_cat,
            self.sdir + "clf_reports.csv",
            indi_suffix=indi_suffix_cat,
            overwrite=True,
        )

        ##############################
        ## ROC-AUC
        ##############################
        roc_aucs_cat = [self.roc_auc_cv_mean, self.roc_auc_cv_std] + self.roc_aucs_folds
        indi_suffix_cat = (
            ["{}-fold CV mean".format(self.num_folds)]
            + ["{}-fold CV std.".format(self.num_folds)]
            + ["fold#{}".format(i) for i in range(self.num_folds)]
        )
        utils.general.save_listed_scalars_as_csv(
            roc_aucs_cat,
            self.sdir + "roc-auc.csv",
            column_name="ROC-AUC",
            indi_suffix=indi_suffix_cat,
            overwrite=True,
        )


def take_mean_and_std(obj_list, n_round=3):
    arr = np.array(obj_list)
    return arr.mean(axis=0).round(n_round), arr.std(axis=0).round(n_round)


def summarize_dfs(dfs_list, method="sum", n_round=3):
    df_zero = 0 * dfs_list[0].copy()  # get the table format

    if method == "sum":
        return df_zero + np.array(dfs_list).sum(axis=0).round(n_round)

    if method == "mean":
        arr = np.array(dfs_list)
        df_mean = arr.mean(axis=0).round(n_round)
        df_std = arr.std(axis=0).round(n_round)
        return df_zero + df_mean, df_zero + df_std


if __name__ == "__main__":
    """A minimal example for using the Reporter class."""
    import numpy as np
    import scipy

    reporter = Reporter("/tmp/")
    N_FOLDS = 5
    bs = 64
    labels = ["class_0", "class_1"]
    n_classes = len(labels)

    for i_fold in range(N_FOLDS):
        ## Test Step in the k-fold CV loop
        true_class_tes = np.random.randint(0, n_classes, size=bs)
        pred_class_tes = np.random.randint(0, n_classes, size=bs)
        pred_proba_tes = scipy.special.softmax(np.random.rand(bs, n_classes), axis=-1)
        reporter.calc_metrics(
            true_class_tes, pred_class_tes, pred_proba_tes, labels=labels, i_fold=i_fold
        )

    reporter.summarize()
    meta_dict = dict(aaa=1, bbb=2)  # if you have additional meta data to save
    reporter.save(meta_dict)

    ## EOF
