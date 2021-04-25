#!/usr/bin/env python
import argparse
from sklearn.mixture import GaussianMixture
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['font.size'] = 20
plt.rcParams["figure.figsize"] = (20, 20)
from mpl_toolkits.mplot3d import Axes3D

import utils.general as ug
import utils.semi_ripple as us
from utils.EDA.plots_3d_scatter import plot_3d_scatter


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--n_mouse", default='01', choices=['01', '02', '03', '04', '05'], \
                help=" ")
args = ap.parse_args()


## Fixes random seed
ug.fix_seed(seed=42)


## FPATHs
LPATH_HIPPO_LFP_NPY_LIST = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
LPATHS_MOUSE = ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse)[1]


## Loads
lfps, _rips_df = us.load_lfps_rips_sec(LPATHS_MOUSE,
                                       rip_sec_ver='ripple_candi_1kHz_pkl/with_props'
                                       )
len_rips_df = np.array([len(_rip) for _rip in _rips_df]) 
rips_df = pd.concat(_rips_df)


## GMM Clustering
ftr1, ftr2, ftr3 = 'ln(duration_ms)', 'mean ln(MEP magni. / SD)', 'ln(ripple peak magni. / SD)'
rips_df = rips_df[[ftr1, ftr2, ftr3]]
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(rips_df)
cls1_proba = gmm.predict_proba(rips_df)[:, 1]
are_cls1 = (cls1_proba >= .5) if gmm.means_[0,1] < gmm.means_[1,1] else (cls1_proba < .5)


## Prepares sparse Data Frame for visualization
perc = .2 if args.n_mouse == '01' else .05
N = int(len(rips_df) * perc / 100)
_indi_sparse = np.random.permutation(len(rips_df))[:N]
indi_sparse = np.zeros(len(rips_df))
indi_sparse[_indi_sparse] = 1
indi_sparse = indi_sparse.astype(bool)


## Defines clusters
cls0_sparse_rips_df = rips_df[~are_cls1 & indi_sparse]
cls1_sparse_rips_df = rips_df[are_cls1 & indi_sparse]



## Plots
spath_mp4 = ug.mk_spath('mouse_{n}.mp4'.format(n=args.n_mouse), makedirs=True)
plot_3d_scatter(cls0_sparse_rips_df,
                ftr1,
                ftr2,
                ftr3,
                cls0_label='Cluster F',
                cls1_sparse_df=cls1_sparse_rips_df,
                cls1_label='Cluster T',
                spath_png=None,
                spath_mp4=spath_mp4,
                title='Mouse #{}\nSparsity: {}%\n'.format(args.n_mouse, perc),
                theta=165,
                phi=3,
                size=10,
                )


## EOF
