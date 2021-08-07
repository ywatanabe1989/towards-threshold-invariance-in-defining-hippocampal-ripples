#!/usr/bin/env python
import argparse
import gc
import re
import sys
from collections import Counter
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from cleanlab.latent_estimation import \
    estimate_confident_joint_and_cv_pred_proba
from cleanlab.pruning import get_noise_indices
from sklearn.model_selection import StratifiedKFold

sys.path.append(".")
import utils
from models.ResNet1D.CleanLabelResNet1D import CleanLabelResNet1D

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="05", choices=["01", "02", "03", "04", "05"], help=" "
)
ap.add_argument("-i", "--include", action="store_true", default=False, help=" ")
args = ap.parse_args()


## Defines dataset
dataset_key = (
    "D{}+".format(args.n_mouse) if args.include else "D{}-".format(args.n_mouse)
)

## Sets tee
spath = utils.general.mk_spath(f"log/{args.n_mouse}/{dataset_key}")
sys.stdout, sys.stderr = utils.general.tee(sys, sdir=spath + "/")


## Fixes random seeds
utils.general.fix_seeds(seed=42, np=np, torch=torch)


## FPATHs
print("\ndataset_key: {}\n".format(dataset_key))
MICE_TO_LOAD = (
    args.n_mouse
    if args.include
    else utils.general.pop_keys(["01", "02", "03", "04", "05"], args.n_mouse)
)
LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.pj.load.get_hipp_lfp_fpaths(MICE_TO_LOAD)
pprint(LPATH_HIPPO_LFP_NPY_LIST_MICE)

SDIR_CLEANLAB = "./data/okada/cleanlab_results/{}/".format(dataset_key)

################################################################################
## Loads
################################################################################
# lfps, rips_df_list_GMM_labeled = utils.pj.load.lfps_rips_sec(
#     LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="GMM_labeled/D{}-".format(args.n_mouse)
# )
rips_df_list_GMM_labeled = utils.pj.load.rips_sec(
    LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="GMM_labeled/{}".format(dataset_key)
)
"""
D0?- is softlinked to D0?+. For example, GMM_labeled/D05- is identical with GMM_labeled/D01+, GMM_labeled/D02+, GMM_labeled/D03+, and GMM_labeled/D04+.
"""
# del lfps
# lfps, rips_df_list_isolated = utils.pj.load.lfps_rips_sec(
#     LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="isolated"
# )  # includes isolated LFP during each ripple candidate
# del lfps
rips_df_list_isolated = utils.pj.load.rips_sec(
    LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="isolated"
)  # includes isolated LFP during each ripple candidate


mice_label = np.array(
    [
        int(l.split("./data/okada/")[1].split("/day")[0])
        for l in LPATH_HIPPO_LFP_NPY_LIST_MICE
    ]
)
################################################################################
## Organizes rips_df
################################################################################
len_rips = [
    len(_rips_df_tt) for _rips_df_tt in rips_df_list_GMM_labeled
]  # to save at last
mice_label = np.hstack(
    [[m for _ in range(rep)] for m, rep in zip(mice_label, len_rips)]
)
rips_df_list_GMM_labeled = pd.concat(rips_df_list_GMM_labeled)
rips_df_list_isolated = pd.concat(rips_df_list_isolated)
rips_df = pd.concat([rips_df_list_GMM_labeled, rips_df_list_isolated], axis=1)  # concat
# Delete unnecessary columns
rips_df = rips_df.loc[:, ~rips_df.columns.duplicated()]  # delete duplicated columns
rips_df = rips_df[["start_sec", "end_sec", "are_ripple_GMM", "isolated"]]
# 'start_sec', 'end_sec',
# 'ln(duration_ms)', 'ln(mean MEP magni. / SD)', 'ln(ripple peak magni. / SD)',
# 'are_ripple_GMM'


################################################################################
## Data
################################################################################
X_all = np.vstack(rips_df["isolated"])
T_all = np.hstack(rips_df["are_ripple_GMM"]).astype(int)
M_all = mice_label  # alias
del rips_df_list_GMM_labeled, rips_df_list_isolated, rips_df
gc.collect()


################################################################################
## Parameters
################################################################################
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS)
N_CLASSES = len(np.unique(T_all))


################################################################################
## Model
################################################################################
cl_conf = utils.general.load("./models/ResNet1D/CleanLabelResNet1D.yaml")
cl_conf["ResNet1D"]["SEQ_LEN"] = X_all.shape[-1]
model = CleanLabelResNet1D(cl_conf)

################################################################################
## Confident Learning using cleanlab
################################################################################
## https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/latent_estimation.py
## estimate_confident_joint_and_cv_pred_proba()

## Calculates predicted probabilities psx.
reporter = utils.ml.Reporter(sdir=SDIR_CLEANLAB)
psx = np.zeros((len(T_all), N_CLASSES))


X_tra_d, T_tra_d, M_tra_d = dict(), dict(), dict()
X_tes_d, T_tes_d, M_tes_d = dict(), dict(), dict()
tra_d, tes_d = dict(), dict()

for i_fold, (indi_tra, indi_tes) in enumerate(skf.split(X_all, M_all)):
    X_tra, T_tra, M_tra = X_all[indi_tra], T_all[indi_tra], M_all[indi_tra]
    X_tes, T_tes, M_tes = X_all[indi_tes], T_all[indi_tes], M_all[indi_tes]

    ## Under sampling
    T_M_tra = utils.ml.merge_labels(T_tra, M_tra, to_int=False)
    indi = utils.ml.under_sample(T_M_tra)
    indi_tra = indi_tra[indi]
    X_tra, T_tra, M_tra = X_all[indi_tra], T_all[indi_tra], M_all[indi_tra]
    # print(Counter(T_tra))
    # print(Counter(M_tra))

    ## Instantiates a Model
    model = CleanLabelResNet1D(cl_conf)

    ## Training
    model.fit(X_tra, T_tra)

    ## Prediction
    pred_proba_tes_fold = model.predict_proba(X_tes)
    pred_class_tes_fold = pred_proba_tes_fold.argmax(dim=-1)
    T_tes_fold = torch.tensor(T_tes)

    reporter.calc_metrics(
        T_tes_fold,
        pred_class_tes_fold,
        pred_proba_tes_fold,
        labels=cl_conf["ResNet1D"]["LABELS"],
        i_fold=i_fold,
    )
    ## to the buffer
    psx[indi_tes] = pred_proba_tes_fold

## Calculates latent error indice
are_errors = get_noise_indices(
    T_all,
    psx,
    inverse_noise_matrix=None,
    prune_method="prune_by_noise_rate",
    n_jobs=20,
)
error_rate = are_errors.mean()
print("\nLabel Errors Rate:\n{:.3f}\n".format(error_rate))
# print('\nLabel Errors Indice:\n{}\n'.format(are_errors))


## Saves the k-fold CV training
reporter.summarize()
are_ripple_GMM = T_all.astype(bool)
others_dict = {
    "are_errors.npy": are_errors,
    "are_ripple_GMM.npy": are_ripple_GMM,
    "psx_ripple.npy": psx[:, 1],
}
reporter.save(others_dict=others_dict)


################################################################################
## Saves
################################################################################
## Loads original rips_sec
# lfps, rips_df_list = utils.pj.load.lfps_rips_sec(
#     LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="candi_with_props"
# )
# del lfps
rips_df_list = utils.pj.load.rips_sec(
    LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="candi_with_props"
)
len_rips = [len(_rips_df_tt) for _rips_df_tt in rips_df_list]

# Saves
start, end = 0, 0
for i_tt, lfp_path in enumerate(LPATH_HIPPO_LFP_NPY_LIST_MICE):
    end += len_rips[i_tt]
    rips_df_list[i_tt] = rips_df_list[i_tt][["start_sec", "end_sec"]]
    rips_df_list[i_tt]["are_ripple_GMM"] = are_ripple_GMM[start:end]
    rips_df_list[i_tt]["psx_ripple"] = psx[:, 1][start:end]
    rips_df_list[i_tt]["are_errors"] = are_errors[start:end]
    spath = utils.pj.path_converters.LFP_to_ripples(
        lfp_path, rip_sec_ver="CNN_labeled/{}".format(dataset_key)
    )
    utils.general.save(rips_df_list[i_tt], spath)
    start = end

## EOF
