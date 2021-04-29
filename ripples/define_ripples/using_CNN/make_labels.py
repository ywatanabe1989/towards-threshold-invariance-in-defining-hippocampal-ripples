#!/usr/bin/env python
import argparse
from cleanlab.latent_estimation import estimate_confident_joint_and_cv_pred_proba
from cleanlab.pruning import get_noise_indices
from sklearn.mixture import GaussianMixture
import sys
sys.path.append('.')
import numpy as np
import pandas as pd

import utils.general as ug
import utils.semi_ripple as us
import utils.path_converters as upcvt
from models.ResNet1D.CleanLabelResNet1D import CleanLabelResNet1D


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
    dataset_key = 'D' + args.n_mouse + '-'
print('Indice of mice to load: {}'.format(N_MICE))

LPATH_HIPPO_LFP_NPY_LIST = ug.read_txt('./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt')
LPATH_HIPPO_LFP_NPY_LIST_MICE = list(np.hstack(
                [ug.search_str_list(LPATH_HIPPO_LFP_NPY_LIST, nm)[1] for nm in N_MICE]
))


## Loads
lfps, rips_df_list_GMM_labeled = us.load_lfps_rips_sec(LPATH_HIPPO_LFP_NPY_LIST_MICE,
                                           rip_sec_ver='GMM_labeled/{}'.format(dataset_key)
                                           ) # includes labels using GMM on the dataset
_, rips_df_list_isolated = us.load_lfps_rips_sec(LPATH_HIPPO_LFP_NPY_LIST_MICE,
                                           rip_sec_ver='isolated'
                                           ) # includes isolated LFP during each ripple candidate

## Organizes rips_df
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


## Data
X_all = np.vstack(rips_df['isolated'])
T_all = np.hstack(rips_df['are_ripple_GMM'])


## Parameters
SAMP_RATE = ug.get_samp_rate_int_from_fpath(LPATH_HIPPO_LFP_NPY_LIST_MICE[0])
LABELS = ['nonRipple', 'Ripple']
N_CLASSES = len(labels)
BS = 64
N_CHS = 1
SEQ_LEN = X_all.shape[-1]


## Model
model_conf = ug.load('./models/ResNet1D/CleanLabelResNet1D.yaml')
model_conf['labels'] = LABELS
model_conf['n_chs'] = N_CHS
model_conf['seq_len'] = SEQ_LEN
model_conf['lr'] = 1e-3
model_conf['device'] = 'cuda'

dl_conf = {'batch_size': BS,
           'num_workers': 10,
           }

################################################################################
## Confident Learning using cleanlab
################################################################################    
model = CleanLabelResNet1D(model_conf, dl_conf)
# Compute the confident joint and the n x m predicted probabilities matrix (psx),
# for n examples, m classes. Stop here if all you need is the confident joint.
confident_joint, psx = estimate_confident_joint_and_cv_pred_proba(
    X=X_all,
    s=T_all,
    clf=model, # default, you can use any classifier
)

are_errors = get_noise_indices(T_all,
                               psx,
                               inverse_noise_matrix=None,
                               prune_method='prune_by_noise_rate',
                               n_jobs=20,
                               )

print('\nLabel Errors Indice:\n{}\n'.format(are_errors))


## EOF
