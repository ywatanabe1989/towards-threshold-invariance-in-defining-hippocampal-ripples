#!/usr/bin/env python
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.mixture import GaussianMixture

sys.path.append(".")
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="03", choices=["01", "02", "03", "04", "05"], help=" "
)
args = ap.parse_args()


## Fixes random seed
utils.general.fix_seeds(seed=42, np=np)


## FPATHs
LPATH_HIPPO_LFP_NPY_LIST_MOUSE = utils.pj.load.get_hipp_lfp_fpaths(args.n_mouse)
dataset_key = "D" + args.n_mouse + "+"

## Loads
lfps, rips_df_list = utils.pj.load.lfps_rips_sec(
    LPATH_HIPPO_LFP_NPY_LIST_MOUSE, rip_sec_ver="candi_with_props"
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


## Gets the optimal number of clustering from the pre-experiment
elbow_df = pd.read_csv(
    "./ripples/define_ripples/using_GMM/estimates_the_optimal_n_clusters/elbow.csv"
)
n_optimal_clusters = elbow_df["mouse_#{}_n_optimal".format(args.n_mouse)][0]

## GMM Clustering
gmm = GaussianMixture(n_components=n_optimal_clusters, covariance_type="full")
gmm.fit(rips_df)

MEP_DIM = 1
if n_optimal_clusters == 2:
    i_MEP_lowest_cluster = np.argmin(
        [gmm.means_[0, MEP_DIM], gmm.means_[1, MEP_DIM]]
    )  # T_ripple
    pred_proba_ripple_GMM = gmm.predict_proba(rips_df)[:, i_MEP_lowest_cluster]
    are_ripple_GMM = pred_proba_ripple_GMM >= 0.5
    are_the_med_MEP_GMM = np.nan * are_ripple_GMM


if n_optimal_clusters == 3:
    i_MEP_lowest_cluster = np.argmin(gmm.means_[:, MEP_DIM])  # T_ripple
    i_MEP_highest_cluster = np.argmax(gmm.means_[:, MEP_DIM])  # F_ripple
    i_MEP_med_cluster = list({0, 1, 2} - {i_MEP_lowest_cluster, i_MEP_highest_cluster})[
        0
    ]

    assert (i_MEP_lowest_cluster != i_MEP_med_cluster) & (
        i_MEP_med_cluster != i_MEP_highest_cluster
    )

    pred_proba_ripple_GMM = gmm.predict_proba(rips_df)[:, i_MEP_lowest_cluster]
    are_ripple_GMM = pred_proba_ripple_GMM >= 0.5

    pred_proba_the_med_MEP_GMM = gmm.predict_proba(rips_df)[:, i_MEP_med_cluster]
    are_the_med_MEP_GMM = pred_proba_the_med_MEP_GMM >= 0.5


## Appends the GMM's predictions on original rips_df_list
start, end = 0, 0
for i_tt in range(len(rips_df_list)):
    end += len_rips[i_tt]
    rips_df_list[i_tt]["are_ripple_GMM"] = are_ripple_GMM[start:end]
    rips_df_list[i_tt]["are_the_med_MEP_GMM"] = are_the_med_MEP_GMM[start:end]
    start = end


## Saves
for i_tt, lfp_path in enumerate(LPATH_HIPPO_LFP_NPY_LIST_MOUSE):
    spath = utils.pj.path_converters.LFP_to_ripples(
        lfp_path, rip_sec_ver="GMM_labeled/{}".format(dataset_key)
    )
    utils.general.save(rips_df_list[i_tt], spath)


## EOF
