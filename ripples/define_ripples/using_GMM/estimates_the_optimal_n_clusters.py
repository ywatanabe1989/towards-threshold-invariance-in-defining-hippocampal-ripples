#!/usr/bin/env python
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.mixture import GaussianMixture

sys.path.append(".")
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="01", choices=["01", "02", "03", "04", "05"], help=" "
)
ap.add_argument("-i", "--include", action="store_true", default=False, help=" ")
args = ap.parse_args()


## Sets tee
sys.stdout, sys.stderr = utils.general.tee(sys)


## Configure matplotlib
utils.plt.configure_mpl(plt)


## Fixes random seed
utils.general.fix_seeds(seed=42, np=np)


## Functions
def estimates_the_optimal_n_clusters_of_GMM(X, show=True, spath=None, n_rep=5):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    from sklearn.mixture import GaussianMixture

    n_clusters = np.arange(2, 6)

    scores_mean = []
    scores_std = []

    for n in n_clusters:
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

    if show:
        fig, ax = plt.subplots()
        ax.errorbar(n_clusters, scores_mean, yerr=scores_std)
        ax.plot(n_clusters, scores_mean)
        score_str = "Calinski Harabasz Score"
        ax.set_title(score_str)
        ax.set_xticks(n_clusters)
        ax.set_xlabel("N. of clusters")
        ax.set_ylabel("{} (mean +/- std.; n={})".format(score_str, n_rep))
        if spath is not None:
            utils.general.save(plt, spath)
        else:
            fig.show()

    return n_clusters[np.argmax(scores_mean)]


## FPATHs
N_MICE_CANDIDATES = ["01", "02", "03", "04", "05"]
i_mouse_tgt = utils.general.grep(N_MICE_CANDIDATES, args.n_mouse)[0][0]
if args.include:
    N_MICE = [args.n_mouse]
    dataset_key = "D" + args.n_mouse + "+"
if not args.include:
    N_MICE = N_MICE_CANDIDATES.copy()
    N_MICE.pop(i_mouse_tgt)
    dataset_key = "D" + args.n_mouse + "-"


LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
    "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
)
LPATH_HIPPO_LFP_NPY_LIST_MICE = list(
    np.hstack([utils.general.grep(LPATH_HIPPO_LFP_NPY_LIST, nm)[1] for nm in N_MICE])
)
print("Indice of mice to load: {}".format(N_MICE))
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
spath = "elbow_method_{}_n_mouse_{}.png".format(dataset_key, args.n_mouse)
n_optimal = estimates_the_optimal_n_clusters_of_GMM(X, show=True, spath=spath)
n_optimal_df = pd.DataFrame(
    {"The Optimal # of Clusters for GMM": n_optimal}, index=np.arange(1)
)
print("\nOptimal Number of Clusters of GMM is {}".format(n_optimal))


## Saves
utils.general.save(n_optimal_df, "optimal_n_clusters_GMM_{}.csv".format(dataset_key))

## EOF
