#!/usr/bin/env python

import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

sys.path.append(".")
import utils

## Sets tee
sys.stdout, sys.stderr = utils.general.tee(sys)


## Configure matplotlib
utils.plt.configure_mpl(
    plt,
    figsize=(18.1, 4.0),
    fontsize=7,
    tick_size=0.8,
    tick_width=0.2,
    hide_spines=True,
)


## Fixes random seed
utils.general.fix_seeds(seed=42, np=np)


## Functions
def elbow_calinski_harabasz_for_the_optimal_n_clusters_of_GMM(
    X, n_rep=5, n_clusters_candi=np.arange(2, 6)
):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    from sklearn.mixture import GaussianMixture

    scores_mean = []
    scores_std = []

    for n in tqdm(n_clusters_candi):
        score_n_cluster = []
        for i_rep in range(n_rep):
            gmm = GaussianMixture(n_components=n, n_init=2, covariance_type="full").fit(
                X
            )
            labels = gmm.predict(X)
            score = metrics.calinski_harabasz_score(X, labels)
            score_n_cluster.append(score)
        scores_mean.append(np.mean(score_n_cluster))
        scores_std.append(np.std(score_n_cluster))

    return scores_mean, scores_std


N_MICE = ["01", "02", "03", "04", "05"]
fig, axes = plt.subplots(1, len(N_MICE), sharey=True)
elbow_dict = {}

for i_mouse, n_mouse in enumerate(N_MICE):

    ## FPATHs
    dataset_key = "D{}+".format(n_mouse)
    LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.pj.load.get_hipp_lfp_fpaths(n_mouse)
    print("Indice of mice to load: {}".format(n_mouse))
    print(len(LPATH_HIPPO_LFP_NPY_LIST_MICE))

    ## Loads
    lfps, rips_df_list = utils.pj.load.lfps_rips_sec(
        LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="candi_with_props"
    )
    len_rips = [len(_rips_df_tt) for _rips_df_tt in rips_df_list]
    rips_df = pd.concat(rips_df_list)
    ftr1, ftr2, ftr3 = (
        "ln(duration_ms)",
        "ln(mean MEP magni. / SD)",
        "ln(ripple peak magni. / SD)",
    )
    rips_df = rips_df[[ftr1, ftr2, ftr3]]
    keys_to_remain = ["start_sec", "end_sec", ftr1, ftr2, ftr3]
    for i_rips in range(len(rips_df_list)):
        rips_df_list[i_rips] = rips_df_list[i_rips][keys_to_remain]

    ################################################################################
    ## Finds the optimal number of clusters
    ################################################################################
    # https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
    X = np.array(rips_df, dtype=np.float32)
    X = X[:1000]

    n_rep = 5
    n_clusters_candi = np.arange(2, 6)
    # scores_mean, scores_std = estimates_the_optimal_n_clusters_of_GMM(
    scores_mean, scores_std = elbow_calinski_harabasz_for_the_optimal_n_clusters_of_GMM(
        X,
        n_rep=n_rep,
        n_clusters_candi=n_clusters_candi,
    )
    n_optimal = n_clusters_candi[np.argmax(scores_mean)]

    ## for saving
    elbow_dict.update(
        {
            "mouse_#{}_scores_mean".format(n_mouse): scores_mean,
            "mouse_#{}_scores_std".format(n_mouse): scores_std,
            "mouse_#{}_n_optimal".format(n_mouse): [
                n_optimal for _ in range(len(scores_std))
            ],
        }
    )

    ## Plots
    ax = axes[i_mouse]
    ax = utils.plt.ax_extend(ax, 1, 0.75)
    ax = utils.plt.ax_set_position(fig, ax, 0.2, 2, dragv=True)
    ax.errorbar(n_clusters_candi, scores_mean, yerr=scores_std)
    ax.plot(n_clusters_candi, scores_mean)
    ax.set_title("Mouse #{}".format(n_mouse))
    ax.set_xticks(n_clusters_candi)
    fig.supxlabel("# of clusters")
    # fig.supylabel("Calinski Harabasz Score\n(mean +/- std.; n={})".format(n_rep))
    fig.supylabel("Calinski\nHarabasz\nscore")
    tick_spacing = (ax.get_ylim()[1] - ax.get_ylim()[0]) // 4
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

## Saves
# fig.show()
utils.general.save(fig, "elbow.png")

utils.general.save(pd.DataFrame(elbow_dict), "elbow.csv")

## EOF
