#!/usr/bin/env python
import argparse
from sklearn.mixture import GaussianMixture
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.general as ug
import utils.semi_ripple as us
import utils.path_converters as upcvt



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--n_mouse", default='01', choices=['01', '02', '03', '04', '05'], \
                help=" ")
args = ap.parse_args()


## Configure Matplotlib
ug.configure_mpl(plt)


## Fixes random seed
ug.fix_seed(seed=42)


## FPATHs
LPATH_HIPPO_LFP_NPY_LIST = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
LPATH_HIPPO_LFP_NPY_LIST_MOUSE = ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse)[1]


## Loads
lfps, rips_df_list = us.load_lfps_rips_sec(LPATH_HIPPO_LFP_NPY_LIST_MOUSE,
                                       rip_sec_ver='with_props'
                                       )
len_rips = [len(_rips_df_tt) for _rips_df_tt in rips_df_list]
rips_df = pd.concat(rips_df_list)
ftr1, ftr2, ftr3 = 'ln(duration_ms)', 'mean ln(MEP magni. / SD)', 'ln(ripple peak magni. / SD)'
rips_df = rips_df[[ftr1, ftr2, ftr3]]


## GMM Clustering
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(rips_df)
cls1_proba = gmm.predict_proba(rips_df)[:, 1]
are_ripple_GMM = (cls1_proba >= .5) if gmm.means_[0,1] < gmm.means_[1,1] else (cls1_proba < .5)


## Appends the GMM's predictions on original rips_df_list
start, end = 0, 0
for i_tt in range(len(rips_df_list)):
    end += len_rips[i_tt]
    rips_df_list[i_tt]['are_ripple_GMM'] = are_ripple_GMM[start:end]
    start = end


    
## Saves
for i_tt, lfp_path in enumerate(LPATH_HIPPO_LFP_NPY_LIST_MOUSE):
    spath = upcvt.LFP_to_ripple_candi_with_props(lfp_path)
    spath = spath.replace('with_props', 'with_GMM_preds')
    ug.save(rips_df_list[i_tt], spath)


## EOF
