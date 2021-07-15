#!/usr/bin/env python

import argparse
import multiprocessing as mp
import os
import random
import sys
from bisect import bisect_left
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
def _define_X_T_electrode(lfp, rip_sec, **kwargs):
    """
    Defines LFP epochs.
        Tb:
            'n': not ripple including LFP epoch
            's': just one suspicioius ripple including LFP epoch (1 SD < riple peak magnitude < 7 SD)
            'r': just one reasonable ripple including LFP epoch (7 SD <= riple peak magnitude)

    Example:
            lfp, rip_sec = lfps_tra[0], rips_tra[0]
            kwargs = {
                "samp_rate": 1000,
                "window_size_pts": 400,
                "use_fp16": True,
                "use_shuffle": True,
                "use_random_start": False,
            }
            Xb, Tb = define_X_T_electrode(lfp, rip_sec, **kwargs)
    """

    ##############################
    ## Random start
    ##############################
    random_start_pts = (
        random.randint(0, kwargs["window_size_pts"])
        if kwargs["use_random_start"]
        else 0
    )

    ## Fills rip_sec
    rip_filled = utils.pj.fill_rip_sec(lfp, rip_sec, kwargs["samp_rate"])

    ## Slices LFP
    the_4th_last_rip_end_pts = int(rip_filled.iloc[-4]["end_sec"] * kwargs["samp_rate"])
    lfp = lfp[random_start_pts:the_4th_last_rip_end_pts]

    ##############################
    ## Splits LFP into segments
    ##############################
    segs = skimage.util.view_as_windows(
        lfp,
        window_shape=(kwargs["window_size_pts"],),
        step=kwargs["window_size_pts"],
    )
    # time points
    segs_start_pts = (
        np.array(
            [random_start_pts + kwargs["window_size_pts"] * i for i in range(len(segs))]
        )
        + 1e-10
    )
    segs_start_sec = segs_start_pts / kwargs["samp_rate"]
    segs_end_pts = segs_start_pts + kwargs["window_size_pts"]
    segs_end_sec = segs_end_pts / kwargs["samp_rate"]

    ##############################
    # Links sliced LFP and filled ripples
    ##############################
    the_1st_indi = np.array(
        [
            bisect_left(rip_filled["start_sec"].values, segs_start_sec[i]) - 1
            for i in range(len(segs))
        ]
    )  # on the starting points
    the_2nd_indi = the_1st_indi + 1
    the_3rd_indi = the_1st_indi + 2

    the_1st_filled_rips = rip_filled.iloc[the_1st_indi]  # on the starting points
    the_2nd_filled_rips = rip_filled.iloc[the_2nd_indi]
    the_3rd_filled_rips = rip_filled.iloc[the_3rd_indi]

    ##############################
    ## T_electrode
    ##############################
    ## Base conditions
    are_the_1st_ripple_CNN = the_1st_filled_rips["are_ripple_CNN"] == 1.0  # fixme
    are_the_2nd_ripple_CNN = the_2nd_filled_rips["are_ripple_CNN"] == 1.0
    are_the_3rd_ripple_CNN = the_3rd_filled_rips["are_ripple_CNN"] == 1.0

    are_the_1st_over_the_slice_end = segs_end_sec < the_1st_filled_rips["end_sec"]
    are_the_2nd_over_the_slice_end = segs_end_sec < the_2nd_filled_rips["end_sec"]
    are_the_3rd_over_the_slice_end = segs_end_sec < the_3rd_filled_rips["end_sec"]

    the_2nd_ripple_peak_magnitude_SD = np.exp(
        the_2nd_filled_rips["ln(ripple peak magni. / SD)"]
    )

    a_condition_for_s_or_r = np.vstack(
        [
            ~are_the_1st_ripple_CNN,
            ~are_the_1st_over_the_slice_end,
            are_the_2nd_ripple_CNN,
            ~are_the_2nd_over_the_slice_end,
            ~are_the_3rd_ripple_CNN,
            are_the_3rd_over_the_slice_end,
        ]
    ).all(axis=0)

    ## Conditions
    # 'n': not ripple including LFP epoch
    are_n = np.vstack(
        [
            ~are_the_1st_ripple_CNN,
            are_the_1st_over_the_slice_end,
            np.isnan(the_2nd_ripple_peak_magnitude_SD),
        ]
    ).all(axis=0)

    # 's': just one suspicioius ripple including LFP epoch (1 SD < riple peak magnitude < 7 SD)
    are_s = (
        a_condition_for_s_or_r
        & (1 <= the_2nd_ripple_peak_magnitude_SD)
        # & (the_2nd_ripple_peak_magnitude_SD < 7)
        & (
            the_2nd_ripple_peak_magnitude_SD
            < kwargs["lower_SD_thres_for_reasonable_ripple"]
        )
    )

    # 'r': just one reasonable ripple including LFP epoch (7 SD <= riple peak magnitude)
    # are_r = a_condition_for_s_or_r & (7 <= the_2nd_ripple_peak_magnitude_SD)
    are_r = a_condition_for_s_or_r & (
        kwargs["lower_SD_thres_for_reasonable_ripple"]
        <= the_2nd_ripple_peak_magnitude_SD
    )
    ##############################
    ## X_electrode
    ##############################
    segs_n = segs[are_n]
    segs_s = segs[are_s]
    segs_r = segs[are_r]

    X_electrode = np.vstack([segs_n, segs_s, segs_r])
    T_electrode = np.array(
        ["n" for _ in range(len(segs_n))]
        + ["s" for _ in range(len(segs_s))]
        + ["r" for _ in range(len(segs_r))]
    )

    return X_electrode, T_electrode


def _define_X_T_electrode_wrapper(arg_list):
    args, kwargs = arg_list
    return _define_X_T_electrode(*args, **kwargs)


def _define_X_T_electrodes_mp(arg_list):
    p = mp.Pool(mp.cpu_count())
    output = p.map(_define_X_T_electrode_wrapper, arg_list)
    p.close()
    return output


def define_X_T_electrodes_mp_wrapper(lfps, rips_sec, **kwargs):
    arg_list = [((lfps[i], rips_sec[i]), kwargs) for i in range(len(lfps))]

    X_T_mapped = _define_X_T_electrodes_mp(arg_list)  # too big
    X = np.vstack([m[0] for m in X_T_mapped])
    T = np.hstack([m[1] for m in X_T_mapped])

    # import pdb

    # pdb.set_trace()
    indi_b = np.vstack([T == c for c in kwargs["use_classes_str"]]).any(axis=0)
    # indi_b = indi_b.any(axis=0)

    X, T = X[indi_b], T[indi_b]

    return X, T


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


if __name__ == "__main__":
    i_mouse_test = 0
    lfps_tra, rips_tra = utils.pj.load.lfps_rips_tra_or_tes("tra", i_mouse_test)
    lfps_tes, rips_tes = utils.pj.load.lfps_rips_tra_or_tes("tes", i_mouse_test)
    print(len(lfps_tra), len(lfps_tes))

    # lfp, rip_sec = lfps_tra[0], rips_tra[0]
    kwargs = {
        # "use_classes_str": ["n", "s", "r"],
        "use_classes_str": ["n", "r"],
        "samp_rate": 1000,
        "window_size_pts": 400,
        "use_fp16": True,
        "use_shuffle": True,
        "use_random_start": False,
        "lower_SD_thres_for_reasonable_ripple": 7,
    }

    # ## define_X_T_electrode
    # Xb, Tb = _define_X_T_electrode(lfps_tra[0], rips_tra[0], **kwargs)
    # Xb, Tb = _define_X_T_electrode(lfps_tes[0], rips_tes[0], **kwargs)

    ##
    X_tra, T_tra = define_X_T_electrodes_mp_wrapper(lfps_tra, rips_tra, **kwargs)
    X_tes, T_tes = define_X_T_electrodes_mp_wrapper(lfps_tes, rips_tes, **kwargs)

    # dlfiller = DataLoaderFiller()
    # count = 0
    # for i in range(len(dlfiller.lfps)):
    #     count += len(dlfiller.lfps[i])

    print((T_tra == "n").sum())
    print((T_tra == "s").sum())
    print((T_tra == "r").sum())
