#!/usr/bin/env python

import argparse
import os
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import skimage
import torch
import utils
from natsort import natsorted
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset

################################################################################
## Fixes random seeds
################################################################################
utils.general.fix_seeds(seed=42, np=np, torch=torch)


################################################################################
## FPATHs
################################################################################
# lfps, rips_df_list_GMM_labeled = utils.pj.load_mouse_lfp_rip(rip_sec_ver="GMM_labeled")


class DataLoaderFiller:
    def __init__(
        self,
        i_test_mouse=0,
        window_size=400,
        batch_size=64,
        num_workers=0,
        do_under_sampling=True,
        dtype="fp16",
        RANDOM_STATE=42,
        **kwargs,
    ):

        ################################################################################
        ## Fix random seed
        ################################################################################
        self.RANDOM_STATE = RANDOM_STATE
        utils.general.fix_seeds(seed=RANDOM_STATE, random=random, np=np, torch=torch)

        ################################################################################
        ## Attributes
        ################################################################################
        self.i_test_mouse = i_test_mouse
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.do_under_sampling = do_under_sampling
        self.dtype_np = np.float32 if dtype == "fp32" else np.float16
        self.dtype_torch = torch.float32 if dtype == "fp32" else torch.float16

        ################################################################################
        ## Loads dataset and splits_dataset
        ################################################################################

        # tramouse <= D+
        # testmouse <= D-
        self.lfps, self.rips_df, self.i_mice = [], [], []
        # for i_mouse in range(1, 6):
        #     _lfps, _rips_df_list_GMM_labeled = utils.pj.load_mouse_lfp_rip(
        #         mouse_num="0{}".format(i_mouse), rip_sec_ver="GMM_labeled"
        #     )
        #     self.lfps.append(_lfps)
        #     self.rips_df.append(_rips_df_list_GMM_labeled)
        #     self.i_mice.append([i_mouse])

        # i_test_mouse = "01"
        # lfps_tes = lfps.pop(int(i_test_mouse) - 1)
        # lfps_tra = lfps
        # rips_df_tes = rips_df.pop(int(i_test_mouse) - 1)
        # rips_df_tra = rips_df
        # del lfps, rips_df

        # # to tensor
        # _lfps_tra = torch.tensor(np.vstack(lfps_tra))
        # lfps_tes = torch.tensor(np.array(lfps_tes))
        # rips_df_tra = np.array(rips_df_tra)  # not to tensor
        # rips_df_tes = np.array(rips_df_tes)


# dlfiller = DataLoaderFiller()
# count = 0
# for i in range(len(dlfiller.lfps)):
#     count += len(dlfiller.lfps[i])


# def split_n_mice_tra_tes(i_mouse_test=0):
#     """
#     i_mouse_test = 0
#     """
#     N_MICE_CANDIDATES = ["01", "02", "03", "04", "05"]
#     n_mice_tes = [N_MICE_CANDIDATES.pop(i_mouse_test)]
#     n_mice_tra = N_MICE_CANDIDATES
#     return n_mice_tra, n_mice_tes


# def split_npy_list_tra_tes(n_mice_tra, n_mice_tes):
#     LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
#         "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
#     )

#     lpath_hippo_lfp_npy_list_tra = list(
#         np.hstack(
#             [
#                 utils.general.grep(LPATH_HIPPO_LFP_NPY_LIST, nm_tra)[1]
#                 for nm_tra in n_mice_tra
#             ]
#         )
#     )

#     lpath_hippo_lfp_npy_list_tes = list(
#         np.hstack(
#             [
#                 utils.general.grep(LPATH_HIPPO_LFP_NPY_LIST, nm_tes)[1]
#                 for nm_tes in n_mice_tes
#             ]
#         )
#     )
#     return lpath_hippo_lfp_npy_list_tra, lpath_hippo_lfp_npy_list_tes


# i_mouse_test = 0

# n_mice_tra, n_mice_tes = split_n_mice_tra_tes(i_mouse_test=i_mouse_test)
# lpath_hippo_lfp_npy_list_tra, lpath_hippo_lfp_npy_list_tes = split_npy_list_tra_tes(
#     n_mice_tra, n_mice_tes
# )

# ##############################
# ## Training
# ##############################
# def load_lfps_rips_tra_or_tes(
#     lpath_hippo_lfp_npy_list,
#     tra_or_tes_str,
#     i_mouse_test,
# ):

#     if tra_or_tes_str == "tra":
#         dataset_key = "D0{}-".format(str(i_mouse_test + 1))
#     if tra_or_tes_str == "tes":
#         dataset_key = "D0{}+".format(str(i_mouse_test + 1))

#     lfps, rips = utils.pj.load.lfps_rips_sec(
#         lpath_hippo_lfp_npy_list,
#         rip_sec_ver="CNN_labeled/{}".format(dataset_key),
#     )

#     _rips_GMM = utils.pj.load.rips_sec(
#         lpath_hippo_lfp_npy_list,
#         rip_sec_ver="GMM_labeled/{}".format(dataset_key),
#     )  # to take ln(ripple peak magni. / SD)

#     ln_norm_ripple_peak_key = "ln(ripple peak magni. / SD)"
#     for i_rip in range(len(rips)):
#         rips[i_rip][ln_norm_ripple_peak_key] = _rips_GMM[i_rip][ln_norm_ripple_peak_key]
#     del _rips_GMM

#     return lfps, rips


import utils

i_mouse_test = 1
lfps_tra, rips_tra = utils.pj.load.lfps_rips_tra_or_tes("tra", i_mouse_test)
lfps_tes, rips_tes = utils.pj.load.lfps_rips_tra_or_tes("tes", i_mouse_test)
print(len(lfps_tra), len(lfps_tes))

lfps_tra, rips_tra = load_lfps_rips_tra_or_tes(
    lpath_hippo_lfp_npy_list_tra,
    "tra",
    0,
)

lfps_tes, rips_tes = load_lfps_rips_tra_or_tes(
    lpath_hippo_lfp_npy_list_tes,
    "tes",
    0,
)


# dataset_key_tra = "D0{}-".format(str(i_mouse_test + 1))
# lfps_tra, rips_tra = utils.pj.load.lfps_rips_sec(
#     lpath_hippo_lfp_npy_list_tra, rip_sec_ver="CNN_labeled/{}".format(dataset_key_tra)
# )
# _rips_tra_GMM = utils.pj.load.rips_sec(
#     lpath_hippo_lfp_npy_list_tra, rip_sec_ver="GMM_labeled/{}".format(dataset_key_tra)
# )
# key = "ln(ripple peak magni. / SD)"
# for i_rip in range(len(rips_tra)):
#     rips_tra[i_rip][key] = _rips_tra_GMM[i_rip][key]
# del _rips_tra_GMM

# ##############################
# ## Test
# ##############################
# dataset_key_tes = "D0{}+".format(str(i_mouse_test + 1))
# lfps_tes, rips_tes = utils.pj.load.lfps_rips_sec(
#     lpath_hippo_lfp_npy_list_tes, rip_sec_ver="CNN_labeled/{}".format(dataset_key_tes)
# )
# _rips_tes_GMM = utils.pj.load.rips_sec(
#     lpath_hippo_lfp_npy_list_tes, rip_sec_ver="GMM_labeled/{}".format(dataset_key_tes)
# )
# for i_rip in range(len(rips_tes)):
#     rips_tes[i_rip][key] = _rips_tes_GMM[i_rip][key]
# del _rips_tes_GMM

# def load_lfps_rips_mice(n_mice_list, dataset_plus_or_minus):
#     lfps, rips_df, n_mice = [], [], []
#     for n_mouse in n_mice_list:
#         # _lfps, _rips_df_list_GMM_labeled = utils.pj.load_mouse_lfp_rip(
#         #     mouse_num="0{}".format(i_mouse), rip_sec_ver="GMM_labeled"
#         # )
#         _lfps, _rips_df_list_GMM_labeled = utils.pj.load_mouse_lfp_rip(
#             mouse_num=n_mouse,
#             rip_sec_ver="GMM_labeled",
#             dataset_plus_or_minus=dataset_plus_or_minus,
#         )
#         lfps.append(_lfps)
#         rips_df.append(_rips_df_list_GMM_labeled)
#         n_mice.append(n_mouse)
#         # i_mice.append([i_mouse])
#     return lfps, rips_df, n_mice

# lfps_tra, rips_df_tra, n_mice_tra = load_lfps_rips_mice(n_mice_tra, '-')
