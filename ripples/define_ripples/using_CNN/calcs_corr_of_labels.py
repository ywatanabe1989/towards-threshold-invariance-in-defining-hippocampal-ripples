#!/usr/bin/env python
import argparse
import sys

sys.path.append(".")
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="01", choices=["01", "02", "03", "04", "05"], help=" "
)
args = ap.parse_args()


## Sets tee
sys.stdout, sys.stderr = utils.general.tee(sys)


## Configures matplotlib
utils.plt.configure_mpl(
    plt,
    dpi=100,
    figsize=(20, 7),
    fontsize=7,
    legendfontsize="xx-small",
    hide_spines=True,
    tick_size=0.8,
    tick_width=0.2,
)

# utils.plt.configure_mpl(plt)
SDIR = "./ripples/define_ripples/using_CNN/calcs_corr_of_labels/"

fig_label, axes_label = plt.subplots(1, 5)
fig_prob, axes_prob = plt.subplots(1, 5)
for i_mouse, args.n_mouse in enumerate(["01", "02", "03", "04", "05"]):
    ################################################################################
    ## FPATHs
    ################################################################################
    LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.pj.load.get_hipp_lfp_fpaths(args.n_mouse)

    dataset_keys = ["D0{}-".format(i + 1) for i in range(5)]
    dataset_keys[i_mouse] = dataset_keys[int(args.n_mouse) - 1].replace("-", "+")
    print(
        "\n----------------------------------------\nTarget Mouse: {}\nDataset Keys: {}\n".format(
            args.n_mouse, dataset_keys
        )
    )

    ################################################################################
    ## Loads
    ################################################################################
    labels_dict = utils.general.listed_dict(dataset_keys)
    pred_probas_dict = utils.general.listed_dict(dataset_keys)
    for dk in dataset_keys:
        rips_df = utils.pj.load.rips_sec(
            LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="CNN_labeled/{}".format(dk)
        )
        rips_df = pd.concat(rips_df)
        rips_df = utils.pj.invert_ripple_labels(rips_df)
        labels_dict[dk] = rips_df["are_ripple_CNN"]
        pred_probas_dict[dk] = rips_df["psx_ripple"]

    ################################################################################
    ## Label concordance rate
    ################################################################################
    five_diag_arr = np.diag([1 for _ in range(5)]).astype(np.float64)
    label_concordance_rates_df = pd.DataFrame(
        data=five_diag_arr, columns=dataset_keys, index=dataset_keys
    )

    label_concordance_rates_df = label_concordance_rates_df[::-1]

    for i_combi, combi in enumerate(combinations(dataset_keys, 2)):
        concordance_rate = np.round(
            (labels_dict[combi[0]] == labels_dict[combi[1]]).mean(), 2
        )
        label_concordance_rates_df.loc[combi[0], combi[1]] = concordance_rate
    print("\nLabels Concordance Rates:\n{}\n".format(label_concordance_rates_df))

    ax_label = axes_label[i_mouse]
    ax_label.set_aspect(1)

    cbar = False if not args.n_mouse == "05" else False
    ax_label = sns.heatmap(
        label_concordance_rates_df,
        mask=label_concordance_rates_df == 0,
        cmap="Blues",
        annot=True,
        annot_kws={"fontsize": 6},
        ax=ax_label,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
    )
    ax_label.set_title("Mouse #{}".format(args.n_mouse))
    ax_label.set_yticklabels(
        ax_label.get_yticklabels(),
        rotation=60,
        fontdict={"horizontalalignment": "right"},
    )
    ax_label.tick_params(direction="out", pad=0)
    dx = 0.3
    ax_label = utils.plt.ax_set_position(fig_label, ax_label, -2 * dx + dx * i_mouse, 0)

    ################################################################################
    ## Correlation of Predicted Probabilities for Ripples
    ################################################################################
    corr_pred_probas_df = pd.DataFrame(
        data=five_diag_arr, columns=dataset_keys, index=dataset_keys
    )
    for i_combi, combi in enumerate(combinations(dataset_keys, 2)):
        corr = np.round(
            np.corrcoef(pred_probas_dict[combi[0]], pred_probas_dict[combi[1]]), 2
        )[0, 1]
        corr_pred_probas_df.loc[combi[0], combi[1]] = corr

    corr_pred_probas_df = corr_pred_probas_df[::-1]

    print(
        "\nCorrelation Coefficients of Predicted Probabilities for Ripples by CNN:\n{}\n".format(
            corr_pred_probas_df
        )
    )

    # fig_prob, ax_prob = plt.subplots()
    ax_prob = axes_prob[i_mouse]

    ax_prob.set_aspect(1)

    cbar = False if not args.n_mouse == "05" else False
    seismic_r = matplotlib.cm.get_cmap("seismic_r")
    ax_prob = sns.heatmap(
        corr_pred_probas_df,
        mask=label_concordance_rates_df == 0,
        cmap=seismic_r,
        annot=True,
        annot_kws={"fontsize": 6},
        ax=ax_prob,
        vmin=-1.0,
        vmax=1.0,
        cbar=cbar,
    )

    ax_prob.set_title("Mouse #{}".format(args.n_mouse))
    ax_prob.set_yticklabels(
        ax_prob.get_yticklabels(),
        rotation=60,
        fontdict={"horizontalalignment": "right"},
    )
    ax_prob.tick_params(direction="out", pad=0)
    dx = 0.3
    ax_prob = utils.plt.ax_set_position(fig_label, ax_prob, -2 * dx + dx * i_mouse, 0)

"""
fig_label.show()
fig_prob.show()
"""

## Saves
utils.general.save(fig_label, SDIR + "label_concordance_rates.png")
utils.general.save(fig_prob, SDIR + "corr_coef_pred_proba.png")

# python3 ./ripples/define_ripples/using_CNN/calcs_corr_of_labels.py

## EOF
