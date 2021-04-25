#!/usr/bin/env python
import argparse
import sys
sys.path.append('.')
import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['font.size'] = 20
plt.rcParams["figure.figsize"] = (20, 20)
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2

import utils.general as ug
import utils.semi_ripple as us
from utils.EDA.plots_3d_scatter import plot_3d_scatter




ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--n_mouse", default='02', choices=['01', '02', '03', '04', '05'], \
                help=" ")
args = ap.parse_args()

## Parse File Path
LPATH_HIPPO_LFP_NPY_LIST = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
LPATHS_MOUSE = ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse)[1]

## Loads
lfps, _rips_df = us.load_lfps_rips_sec(LPATHS_MOUSE,
                                       rip_sec_ver='ripple_candi_1kHz_pkl/with_props'
                                       )
len_rips_df = np.array([len(_rip) for _rip in _rips_df]) 
rips_df = pd.concat(_rips_df)


# Prepares sparse Data Frame for visualization
perc = .2 if args.n_mouse == '01' else .05
N = int(len(rips_df) * perc / 100)
indi_sparse = np.random.permutation(len(rips_df))[:N]
sparse_rips_df = rips_df.iloc[indi_sparse]


## Plots
theta, phi = 165, 3
ftr1, ftr2, ftr3 = 'ln(duration_ms)', 'mean ln(MEP magni. / SD)', 'ln(ripple peak magni. / SD)'
title = 'Mouse #{} \n\
         Sparsity: {}%\n\
        '.format(args.n_mouse, perc)
plot_3d_scatter(sparse_rips_df, ftr1, ftr2, ftr3,
                cls1_label=None,
                spath_png=None, spath_mp4=None,
                title=title,
                theta=theta, phi=phi, perc=perc)

## EOF
