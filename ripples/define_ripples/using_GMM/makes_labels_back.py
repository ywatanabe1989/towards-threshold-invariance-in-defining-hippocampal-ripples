#!/usr/bin/env python
import argparse
from sklearn.mixture import GaussianMixture
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

# LPATH_HIPPO_LFP_NPY_LIST_MOUSE = ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse)[1]


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


## GMM Clustering
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(rips_df)
cls1_proba = gmm.predict_proba(rips_df)[:, 1]


# print(cls1_proba[:10])
# [0.97624474 0.99284637 0.99984701 0.99918557 0.94821748 0.92253847
#  0.98240508 0.62239122 0.96525563 0.84800725]
are_ripple_GMM = (cls1_proba >= .5) if gmm.means_[0,1] < gmm.means_[1,1] else (cls1_proba < .5) # fixme


## Appends the GMM's predictions on original rips_df_list
start, end = 0, 0
for i_tt in range(len(rips_df_list)):
    end += len_rips[i_tt]
    rips_df_list[i_tt]['are_ripple_GMM'] = are_ripple_GMM[start:end]
    start = end


## Saves
for i_tt, lfp_path in enumerate(LPATH_HIPPO_LFP_NPY_LIST_MICE):
    spath = upcvt.LFP_to_ripples(lfp_path, rip_sec_ver='GMM_labeled/{}'.format(dataset_key)) # fixme
    ug.save(rips_df_list[i_tt], spath)


## EOF
