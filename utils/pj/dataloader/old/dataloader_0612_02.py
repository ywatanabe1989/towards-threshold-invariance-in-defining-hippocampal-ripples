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
## Functions
################################################################################
def mk_rip_peak_magnis_arr(lfps, rips, i_lfp):
    lfp = lfps[i_lfp]
    rip = rips[i_lfp]
    rip_peak_magnis_arr = np.zeros_like(lfp)

    SAMP_RATE = 1000
    for i_r, r in rip.iterrows():
        start_pts = int(r["start_sec"] * SAMP_RATE)
        end_pts = int(r["end_sec"] * SAMP_RATE)
        ripple_peak_magni_SD = np.exp(r["ln(ripple peak magni. / SD)"])
        rip_peak_magnis_arr[start_pts:end_pts] = ripple_peak_magni_SD
    return rip_peak_magnis_arr


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


import utils

i_mouse_test = 0
lfps_tra, rips_tra = utils.pj.load.lfps_rips_tra_or_tes("tra", i_mouse_test)
lfps_tes, rips_tes = utils.pj.load.lfps_rips_tra_or_tes("tes", i_mouse_test)
print(len(lfps_tra), len(lfps_tes))


"""
## Codes ripple peak magnitude as arrays in a different way
rip_peak_magnis_tra = [
    mk_rip_peak_magnis_arr(lfps_tra, rips_tra, i_lfp) for i_lfp in range(len(lfps_tra))
]

rip_peak_magnis_tes = [
    mk_rip_peak_magnis_arr(lfps_tes, rips_tes, i_lfp) for i_lfp in range(len(lfps_tes))
]

utils.general.save(rip_peak_magnis_tra, "/tmp/rip_peak_magnis_tra.pkl")
utils.general.save(rip_peak_magnis_tes, "/tmp/rip_peak_magnis_tes.pkl")
"""

# rip_peak_magnis_tra = utils.general.load("/tmp/rip_peak_magnis_tra.pkl")
# rip_peak_magnis_tes = utils.general.load("/tmp/rip_peak_magnis_tes.pkl")

# import random

# from skimage.util import view_as_windows

# window_size_pts = 400
# lfp = lfps_tra[0]
# rpm = rip_peak_magnis_tra[0]

# ## Slices from random starting time points [pts]
# rand_start = random.randint(0, window_size_pts)
# lfp = lfp[rand_start:]
# rpm = rpm[rand_start:]

# lfp = view_as_windows(lfp, window_shape=window_size_pts, step=window_size_pts)
# rpm = view_as_windows(rpm, window_shape=window_size_pts, step=window_size_pts)
