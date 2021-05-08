#!/usr/bin/env python
import argparse
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import sys
sys.path.append('.')
import numpy as np
import pandas as pd

import utils.general as ug
import utils.semi_ripple as us
import utils.path_converters as upcvt


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--n_mouse", default='01', choices=['01', '02', '03', '04', '05'], \
                help=" ")
ap.add_argument("-i", "--include", action='store_true', default=False,
                help=" ")
args = ap.parse_args()


## Fixes random seed
ug.fix_seeds(seed=42, np=np)


## Functions
def estimates_the_optimal_n_clusters_of_GMM(X, show=False):
    from sklearn.mixture import GaussianMixture
    from sklearn import metrics
    import matplotlib.pyplot as plt
    n_clusters = np.arange(2, 5)
    scores = []
    scores_err = []
    for n in n_clusters:
        gmm = GaussianMixture(n_components=n, n_init=2, covariance_type='full').fit(X) 
        labels = gmm.predict(X)
        score = metrics.calinski_harabasz_score(X, labels)
        scores.append(score)

    if show:
        fig, ax = plt.subplots()
        ax.plot(n_clusters, scores)
        ax.set_title("Scores")
        ax.set_xticks(n_clusters)
        ax.set_xlabel("N. of clusters")
        ax.set_ylabel("Score")
        fig.show()

    return n_clusters[np.argmax(scores)]


## FPATHs
N_MICE_CANDIDATES = ['01', '02', '03', '04', '05']
i_mouse_tgt = ug.search_str_list(N_MICE_CANDIDATES, args.n_mouse)[0][0]
if args.include:
    N_MICE = N_MICE_CANDIDATES[i_mouse_tgt]
    dataset_key = 'D' + args.n_mouse + '+'
if not args.include:
    N_MICE = N_MICE_CANDIDATES.copy()
    N_MICE.pop(i_mouse_tgt)
    dataset_key = 'D' + args.n_mouse + '-' # ug.connect_str_list_with_hyphens(N_MICE)
print('Indice of mice to load: {}'.format(N_MICE))

LPATH_HIPPO_LFP_NPY_LIST = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
LPATH_HIPPO_LFP_NPY_LIST_MICE = list(np.hstack(
                [ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, nm)[1] for nm in N_MICE]
))


## Loads
lfps, rips_df_list = us.load_lfps_rips_sec(LPATH_HIPPO_LFP_NPY_LIST_MICE,
                                           rip_sec_ver='candi_with_props'
                                           )
len_rips = [len(_rips_df_tt) for _rips_df_tt in rips_df_list]
rips_df = pd.concat(rips_df_list)
ftr1, ftr2, ftr3 = 'ln(duration_ms)', 'mean ln(MEP magni. / SD)', 'ln(ripple peak magni. / SD)'
rips_df = rips_df[[ftr1, ftr2, ftr3]]
keys_to_remain = ['start_sec', 'end_sec', ftr1, ftr2, ftr3]
for i_rips in range(len(rips_df_list)):
    rips_df_list[i_rips] = rips_df_list[i_rips][keys_to_remain]


################################################################################
## Finds the optimal number of clusters
################################################################################
# https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
X = np.array(rips_df, dtype=np.float32)
n_optimal = estimates_the_optimal_n_clusters_of_GMM(X)
n_optimal_df = pd.DataFrame({'The Optimal # of Clusters for GMM': n_optimal},
                            index=np.arange(1))
print('\nOptimal Number of Clusters of GMM is {}'.format(n_optimal))


## Saves
ug.save(n_optimal_df, 'optimal_n_clusters_GMM_{}.csv'.format(dataset_key))

## EOF
