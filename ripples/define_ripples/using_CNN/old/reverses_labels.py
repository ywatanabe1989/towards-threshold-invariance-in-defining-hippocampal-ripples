#!/usr/bin/env python
import argparse
import sys; sys.path.append('.')
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


################################################################################
## Fixes random seeds
################################################################################
ug.fix_seeds(seed=42, np=np)


################################################################################
## FPATHs
################################################################################
LPATH_HIPPO_LFP_NPY_LIST = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
# Determines LPATH_HIPPO_LFP_NPY_LIST_MICE and dataset_key
N_MICE_CANDIDATES = ['01', '02', '03', '04', '05']
i_mouse_tgt = ug.search_str_list(N_MICE_CANDIDATES, args.n_mouse)[0][0]
if args.include:
    N_MICE = [args.n_mouse]
    dataset_key = 'D' + args.n_mouse + '+'
if not args.include:
    N_MICE = N_MICE_CANDIDATES.copy()
    N_MICE.pop(i_mouse_tgt)
    dataset_key = 'D' + args.n_mouse + '-'

LPATH_HIPPO_LFP_NPY_LIST_MICE = list(np.hstack(
            [ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, nm)[1] for nm in N_MICE]
))
    
print('Indice of mice to load: {}'.format(N_MICE))
print(len(LPATH_HIPPO_LFP_NPY_LIST_MICE))


################################################################################
## Loads
################################################################################
lfps, rips_df_list_CNN_labeled = us.load_lfps_rips_sec(LPATH_HIPPO_LFP_NPY_LIST_MICE,
                                           rip_sec_ver='CNN_labeled/{}'.format(dataset_key)
                                           ) # includes labels using GMM on the dataset

## Inverse the labels and the predicted probabilities
for i_tt in range(len(rips_df_list_CNN_labeled)):
    rips_df_list_CNN_labeled[i_tt]['are_ripple_CNN'] = \
        1 - rips_df_list_CNN_labeled[i_tt]['are_ripple_CNN']

    rips_df_list_CNN_labeled[i_tt]['pred_probas_ripple_CNN'] = \
        1 - rips_df_list_CNN_labeled[i_tt]['pred_probas_ripple_CNN']


################################################################################
## Saves
################################################################################
## Saves
for i_tt, lfp_path in enumerate(LPATH_HIPPO_LFP_NPY_LIST_MICE):
    spath = upcvt.LFP_to_ripples(lfp_path, rip_sec_ver='CNN_labeled/{}'.format(dataset_key))
    spath = spath.replace('/tt', '/reversed_tt')
    ug.save(rips_df_list_CNN_labeled[i_tt], spath)


## EOF
