#!/usr/bin/env python
import argparse
from cleanlab.latent_estimation import estimate_confident_joint_and_cv_pred_proba
from cleanlab.pruning import get_noise_indices
from sklearn.model_selection import StratifiedKFold # train_test_split,
import sys; sys.path.append('.')
import torch
import numpy as np
import pandas as pd

import utils.general as ug
import utils.semi_ripple as us
import utils.path_converters as upcvt
from utils.Reporter import Reporter
from models.ResNet1D.CleanLabelResNet1D import CleanLabelResNet1D


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
    N_MICE = N_MICE_CANDIDATES[i_mouse_tgt]
    dataset_key = 'D' + args.n_mouse + '+'
    LPATH_HIPPO_LFP_NPY_LIST_MICE = ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, N_MICE)[1]

if not args.include:
    N_MICE = N_MICE_CANDIDATES.copy()
    N_MICE.pop(i_mouse_tgt)
    dataset_key = 'D' + args.n_mouse + '-'
    LPATH_HIPPO_LFP_NPY_LIST_MICE = list(np.hstack(
                [ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, nm)[1] for nm in N_MICE]
    ))
    
print('Indice of mice to load: {}'.format(N_MICE))

SDIR_CLEANLAB = './data/okada/cleanlab_results/{}/'.format(dataset_key)

################################################################################
## Loads
################################################################################
lfps, rips_df_list_GMM_labeled = us.load_lfps_rips_sec(LPATH_HIPPO_LFP_NPY_LIST_MICE,
                                           rip_sec_ver='GMM_labeled/{}'.format(dataset_key)
                                           ) # includes labels using GMM on the dataset
del lfps
lfps, rips_df_list_isolated = us.load_lfps_rips_sec(LPATH_HIPPO_LFP_NPY_LIST_MICE,
                                           rip_sec_ver='isolated'
                                           ) # includes isolated LFP during each ripple candidate
del lfps



################################################################################
## Organizes rips_df
################################################################################
len_rips = [len(_rips_df_tt) for _rips_df_tt in rips_df_list_GMM_labeled]
rips_df_list_GMM_labeled = pd.concat(rips_df_list_GMM_labeled)
rips_df_list_isolated = pd.concat(rips_df_list_isolated)
rips_df = pd.concat([rips_df_list_GMM_labeled, rips_df_list_isolated], axis=1) # concat
# Delete unnecessary columns
rips_df = rips_df.loc[:, ~rips_df.columns.duplicated()] # delete duplicated columns
rips_df = rips_df[['start_sec', 'end_sec', 'are_ripple_GMM', 'isolated']]
# 'start_sec', 'end_sec',
# 'ln(duration_ms)', 'mean ln(MEP magni. / SD)', 'ln(ripple peak magni. / SD)',
# 'are_ripple_GMM'


################################################################################
## Data
################################################################################
X_all = np.vstack(rips_df['isolated'])
T_all = np.hstack(rips_df['are_ripple_GMM']).astype(int)
del rips_df_list_GMM_labeled, rips_df_list_isolated, rips_df
import gc; gc.collect()


################################################################################
## Parameters
################################################################################
# SAMP_RATE = ug.get_samp_rate_int_from_fpath(LPATH_HIPPO_LFP_NPY_LIST_MICE[0])
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS)
N_CLASSES = len(np.unique(T_all))


################################################################################
## Model
################################################################################
cl_conf = ug.load('./models/ResNet1D/CleanLabelResNet1D.yaml')
cl_conf['ResNet1D']['SEQ_LEN'] = X_all.shape[-1]
model = CleanLabelResNet1D(cl_conf)


################################################################################
## Confident Learning using cleanlab
################################################################################    
## https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/latent_estimation.py
## estimate_confident_joint_and_cv_pred_proba()

## Calculates predicted probabilities psx.
reporter = Reporter(sdir=SDIR_CLEANLAB) 
psx = np.zeros((len(T_all), N_CLASSES))
for i_fold, (indi_tra, indi_tes) in enumerate(skf.split(X_all, T_all)):
    X_tra, T_tra = X_all[indi_tra], T_all[indi_tra]
    X_tes, T_tes = X_all[indi_tes], T_all[indi_tes]

    # while len(X_tra) % 4 == 0:
    #     X_tra = X_tra[:-1]
    #     T_tra = T_tra[:-1]        

    # while len(X_tes) % 4 == 0:
    #     X_tes = X_tes[:-1]
    #     T_tes = T_tes[:-1]        

    ## Instantiates a Model
    model = CleanLabelResNet1D(cl_conf)

    ## Training
    model.fit(X_tra, T_tra)

    ## Prediction
    pred_proba_tes_fold = model.predict_proba(X_tes)
    pred_class_tes_fold = pred_proba_tes_fold.argmax(dim=-1)
    T_tes_fold = torch.tensor(T_tes)

    reporter.calc_metrics(T_tes_fold,
                          pred_class_tes_fold,
                          pred_proba_tes_fold,
                          labels=cl_conf['ResNet1D']['LABELS'],
                          i_fold=i_fold,
                          )
    ## to the buffer
    psx[indi_tes] = pred_proba_tes_fold

## Calculates latent error indice
are_errors = get_noise_indices(T_all,
                               psx,
                               inverse_noise_matrix=None,
                               prune_method='prune_by_noise_rate',
                               n_jobs=20,
                               )
# print('\nLabel Errors Indice:\n{}\n'.format(are_errors))




## Cleans labels
cleaned_labels = T_all.copy()
error_rate = are_errors.mean()
print('\nLabel Errors Rate:\n{:.3f}\n'.format(error_rate))
cleaned_labels[are_errors] = 1 - cleaned_labels[are_errors] # cleaning
assert ~np.all(cleaned_labels == T_all)

pred_probas_ripples = psx[:, 1]

## Saves the k-fold CV training
reporter.summarize()
others_dict = {'are_errors.npy': are_errors,
               'cleaned_labels.npy': cleaned_labels,
               'pred_probas_ripples.npy': pred_probas_ripples,
               }
reporter.save(others_dict=others_dict)



################################################################################
## Saves
################################################################################
## Loads original rips_sec
lfps, rips_df_list = us.load_lfps_rips_sec(LPATH_HIPPO_LFP_NPY_LIST_MICE,
                                           rip_sec_ver='candi_with_props'
                                           )
del lfps
len_rips = [len(_rips_df_tt) for _rips_df_tt in rips_df_list]

## Appends the found label errors on rips_df_list
start, end = 0, 0
for i_tt in range(len(rips_df_list)):
    end += len_rips[i_tt]
    rips_df_list[i_tt] = rips_df_list[i_tt][['start_sec', 'end_sec']]
    rips_df_list[i_tt]['are_ripple_CNN'] = cleaned_labels[start:end]
    rips_df_list[i_tt]['pred_probas_ripple_CNN'] = pred_probas_ripples[start:end]    
    start = end

## Saves
for i_tt, lfp_path in enumerate(LPATH_HIPPO_LFP_NPY_LIST_MICE):
    spath = upcvt.LFP_to_ripples(lfp_path, rip_sec_ver='CNN_labeled/{}'.format(dataset_key))
    ug.save(rips_df_list[i_tt], spath)


## EOF
