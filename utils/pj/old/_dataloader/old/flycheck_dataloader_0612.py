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
lfps, rips_df_list_GMM_labeled = utils.pj.load_mouse_lfp_rip(rip_sec_ver="GMM_labeled")


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
        self.lfps, self.rips_df, self.i_mouse= [], []
        for i_mouse in range(1, 5):
            _lfps, _rips_df_list_GMM_labeled = utils.pj.load_mouse_lfp_rip(
                mouse_num="0{}".format(i_mouse), rip_sec_ver="GMM_labeled"
            )
            se.lfps.append(_lfps)
            selg.rips_df.append(_rips_df_list_GMM_labeled)
            
        i_test_mouse = "01"
        lfps_tes = lfps.pop(int(i_test_mouse) - 1)
        lfps_tra = lfps
        rips_df_tes = rips_df.pop(int(i_test_mouse) - 1)
        rips_df_tra = rips_df
        del lfps, rips_df

        # to tensor
        _lfps_tra = torch.tensor(np.vstack(lfps_tra))
        lfps_tes = torch.tensor(np.array(lfps_tes))
        rips_df_tra = np.array(rips_df_tra)  # not to tensor
        rips_df_tes = np.array(rips_df_tes)


dlfiller = DataLoaderFiller()
