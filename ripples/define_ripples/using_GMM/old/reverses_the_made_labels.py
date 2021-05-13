#!/usr/bin/env python
import argparse
# from sklearn.mixture import GaussianMixture
# from sklearn import metrics
import numpy as np
# import pandas as pd

import sys; sys.path.append('.')
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
    N_MICE = [args.n_mouse] # N_MICE_CANDIDATES[i_mouse_tgt]
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


## Saves
for i_tt, lfp_path in enumerate(LPATH_HIPPO_LFP_NPY_LIST_MICE):
    lpath = upcvt.LFP_to_ripples(lfp_path, rip_sec_ver='GMM_labeled/{}'.format(dataset_key))
    rip_sec = ug.load(lpath)
    rip_sec['are_ripple_GMM'] = ~rip_sec['are_ripple_GMM'] # Reversed
    spath = lpath.replace('/tt', '/reversed_tt')
    print(spath)
    # ug.save(rips_df_list[i_tt], spath)


## EOF

