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
## Functions
################################################################################
class DataLoaderFillerFvsT(object):
    """A helper class for providing training and test dataloaders.
    The "fill" method enables re-sampling training data at each epoch.

    kwargs:
        "i_mouse_test":
            the index of mouse for test

        "batch_size":
            default: 64

        "num_workers":
            default: 10

        "do_under_sampling":
            whether or not doing under sampling (default: True)

        "use_classes_str":
            a combination of ["t", "f"], for consistency with DataLoaderFillerNvsR.

        "samp_rate":
            1000

        "window_size_pts":
            window size [points] (or you might call it sliding window size, crop size,
            segment size, epoch size, ...; default: 400).

        "use_random_start":
            True or False. To reduce the sampling bias regarding cropping, "True" turns on
            random starts at the first of every training epoch.

        "lower_SD_thres_for_reasonable_ripple":
            The lower threshold for defining reasonbale ripples [SD] (default: 7).

        "do_under_sampling":
            "True" turns on under sampling on "Training" step after shuffling.

        "is_debug":
            If "True", only a part of the training data are loaded.


    Example:
        dlf = DataLoaderFiller(
            i_mouse_test=0,
            use_classes_str=["t", "f"],
        )
        dl_tra = dlf.fill("Training")
        dl_test = dlf.fill("Test")

        batch = next(iter(dl_tra))

        for epoch in range(10):
            dl_tra = dlf.fill("Training")
            for i_batch, batch in enumerate(dl_tra):
                Xb_tra, Pb_tra = batch
                print(i_batch)
                print(Pb_tra)

        dl_tes = dlf.fill("Test")
        for i_batch, batch in enumerate(dl_tra):
            Xb_tra, Pb_tra = batch
            print(i_batch)
            print(Pb_tra)

    """

    def __init__(
        self,
        i_mouse_test=0,
        batch_size=64,
        num_workers=10,
        do_under_sampling=True,
        is_debug=False,
        RANDOM_STATE=42,
        **kwargs,
    ):

        ## Fixes random seeds
        utils.general.fix_seeds(seed=42, np=np, torch=torch)

        ## Merges kwargs
        _kwargs = dict(
            i_mouse_test=i_mouse_test,
            batch_size=batch_size,
            num_workers=num_workers,
            do_under_sampling=do_under_sampling,
            RANDOM_STATE=RANDOM_STATE,
        )
        _kwargs.update(kwargs)
        self.kwargs = _kwargs

        ################################################################################
        ## Fixes random seed
        ################################################################################
        utils.general.fix_seeds(
            seed=self.kwargs["RANDOM_STATE"], random=random, np=np, torch=torch
        )

        ################################################################################
        ## Loads dataset and splits_dataset
        ################################################################################
        self.lfps_tra, self.rips_tra = utils.pj.load.lfps_rips_tra_or_tes(
            "tra",
            self.kwargs["i_mouse_test"],
            is_debug=is_debug,
        )
        self.lfps_tes, self.rips_tes = utils.pj.load.lfps_rips_tra_or_tes(
            "tes",
            self.kwargs["i_mouse_test"],
            is_debug=is_debug,
        )

        self.is_already_filled_test = False

    def fill(self, step):
        """
        step is eather "Training" or "Test."
        """
        if (step == "Test") & self.is_already_filled_test:
            dl = deepcopy(self.dl_test_back)
            return dl

        ## Switches lfps and rips in depending on step
        if step == "Training":
            lfps = self.lfps_tra
            rips = self.rips_tra

        if step == "Test":
            lfps = self.lfps_tes
            rips = self.rips_tes

        ## To chunked and labeled surpervised data
        X, T = define_X_T_electrodes_mp_and_organize_them(
            lfps, rips, step, **self.kwargs
        )

        dl = DataLoader(
            TensorDataset(*[torch.tensor(d) for d in [X, T]]),
            batch_size=self.kwargs["batch_size"],
            # shuffle=True if step == "Training" else False,
            shuffle=True,
            num_workers=self.kwargs["num_workers"],
            drop_last=True,
        )
        """
        # Even when step != "Training", to disable the following warning, data were shuffled.
        /usr/local/lib64/python3.8/site-packages/sklearn/metrics/_classification.py:1850: UserWarning: y_pred contains classes not in y_true
          warnings.warn('y_pred contains classes not in y_true')
         """

        if step == "Test":
            self.is_already_filled_test = True
            self.dl_test_back = deepcopy(dl)

        return dl


def _define_X_T_electrode(
    lfp,
    rip_sec,
    step,
    lower_SD_thres_for_reasonable_ripple=7,
    samp_rate=1000,
    window_size_pts=400,
    use_random_start=True,
    **kwargs,
):
    """
    Crops LFP signals into segments (the sequence length is kwargs['window_size_pts'] [point]).
    Also, each segment is allocated ripple peak power ['uV'] (P) as a target label.
    When kwargs['step'] is 'test', under sampling and shuffle is not conducted.


    args:
        step:
            'Training' or 'Test' is permitted.

    kwargs:
        "use_classes_str":
            a combination of ["t", "f"], for consistency with DataLoaderFillerNvsR.

        "samp_rate":
            1000

        "window_size_pts":
            window size [points] (or you might call it sliding window size, crop size,
            segment size, epoch size, ...).

        "use_random_start":
            True or False. To reduce the sampling bias regarding cropping, "True" turns on
            random starts at the first of every training epoch.

        "lower_SD_thres_for_reasonable_ripple":
            The lower threshold for defining reasonbale ripples [SD] (default: 7).

        "do_under_sampling":
            "True" turns on under sampling on "Training" step after shuffling.

    Returns:
        X:
            Cropped LFP signals [uV].

        T:
            True label as a Ripple labeled by CNN on Confident Learning after GMM clustering.
            0: False ripple (labeled by CNN)
            1: True ripple (labeled by CNN)

    Example:
        i_mouse_test = 0
        lfps, rips = utils.pj.load.lfps_rips_tra_or_tes(
            "tra",
            i_mouse_test,
            is_debug=True,
        )

        for i in range(len(lfps)):
            lfp, rip = lfps[i], rips[i]
            kwargs = {
                # "use_classes_str": ["t", "f"],
                "use_classes_str": ["f", "t"],
                "lower_SD_thres_for_reasonable_ripple": 1,
                "do_under_sampling": True,
            }
            X_el_tra, T_el_tra = _define_X_T_electrode(lfp, rip, "Training", **kwargs)
            print(np.unique(T_el_tra, return_counts=True))
    """

    ## Check an argument
    assert (step == "Training") or (step == "Test")

    ## Merges kwargs
    _kwargs = dict(
        lower_SD_thres_for_reasonable_ripple=lower_SD_thres_for_reasonable_ripple,
        samp_rate=samp_rate,
        window_size_pts=window_size_pts,
        use_random_start=use_random_start,
    )

    _kwargs.update(kwargs)
    kwargs = _kwargs

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
    are_the_1st_ripple_base = the_1st_filled_rips["are_ripple_CNN"].isna()  # 198641
    are_the_2nd_ripple_CNN = the_2nd_filled_rips["are_ripple_CNN"] == 1.0
    are_the_3rd_ripple_base = the_3rd_filled_rips["are_ripple_CNN"].isna()

    are_the_1st_over_the_slice_end = segs_end_sec < the_1st_filled_rips["end_sec"]
    are_the_2nd_over_the_slice_end = segs_end_sec < the_2nd_filled_rips["end_sec"]
    are_the_3rd_over_the_slice_end = segs_end_sec < the_3rd_filled_rips["end_sec"]

    the_2nd_ripple_peak_magnitude_SD = np.exp(
        the_2nd_filled_rips["ln(ripple peak magni. / SD)"]
    )

    ##############################
    ## Conditions
    ##############################
    are_t = np.vstack(
        [
            are_the_1st_ripple_base,
            ~are_the_1st_over_the_slice_end,
            are_the_2nd_ripple_CNN,
            ~are_the_2nd_over_the_slice_end,
            are_the_3rd_ripple_base,
            are_the_3rd_over_the_slice_end,
        ]
    ).all(axis=0)

    are_f = np.vstack(
        [
            are_the_1st_ripple_base,
            ~are_the_1st_over_the_slice_end,
            ~are_the_2nd_ripple_CNN,
            ~are_the_2nd_over_the_slice_end,
            are_the_3rd_ripple_base,
            are_the_3rd_over_the_slice_end,
        ]
    ).all(axis=0)


    ###################################
    ## Allocates labels, T
    ###################################
    segs_t = segs[are_t]
    segs_f = segs[are_f]

    X_el = np.vstack([segs_t, segs_f])

    P_t = the_2nd_ripple_peak_magnitude_SD[are_t]
    P_f = the_2nd_ripple_peak_magnitude_SD[are_f]
    P_el = np.hstack([P_t, P_f])

    # Converts the dtype of P_el float16
    P_el = P_el.astype(np.float16)

    # Target labels to find only necessary segments
    T_el = np.array(
        ["t" for _ in range(len(segs_t))] + ["f" for _ in range(len(segs_f))]
    )

    # Excludes unnecessary segments depending on kwargs
    indi = np.vstack([T_el == c for c in kwargs["use_classes_str"]]).any(axis=0)
    X_el, T_el = X_el[indi], T_el[indi]

    ## Shuffle within electrode for under sampling
    if step == "Training":
        X_el, T_el = shuffle(X_el, T_el)

    ##################
    # Under Sampling #
    ##################
    if kwargs["do_under_sampling"] & (step == "Training"):
        N_min = np.min([(T_el == c).sum() for c in kwargs["use_classes_str"]])
        indi_classes = [np.where(T_el == c)[0] for c in kwargs["use_classes_str"]]
        indi_pick = np.hstack([ic[:N_min] for ic in indi_classes])
        X_el, T_el = X_el[indi_pick], T_el[indi_pick]

    ################################################################################

    for i_k, k in enumerate(kwargs["use_classes_str"]):
        T_el[T_el == k] = i_k

    T_el = T_el.astype(int)  # OK

    return X_el, T_el


def _define_X_T_electrode_wrapper(arg_list):
    args, kwargs = arg_list
    return _define_X_T_electrode(*args, **kwargs)


def _define_X_T_electrodes_mp(arg_list):
    p = mp.Pool(mp.cpu_count())
    output = p.map(_define_X_T_electrode_wrapper, arg_list)
    p.close()
    return output


def define_X_T_electrodes_mp_and_organize_them(lfps, rips_sec, step, **kwargs):
    arg_list = [((lfps[i], rips_sec[i], step), kwargs) for i in range(len(lfps))]

    X_T_mapped = _define_X_T_electrodes_mp(arg_list)

    X = np.vstack([m[0] for m in X_T_mapped])
    T = np.hstack([m[1] for m in X_T_mapped])

    return X, T


if __name__ == "__main__":
    DL_CONF = {
        "batch_size": 1024,
        "num_workers": 10,
        "do_under_sampling": True,
        "use_classes_str": ["f", "t"],
        # "use_classes_str": ["t", "f"],
        "samp_rate": 1000,
        "window_size_pts": 400,
        "use_random_start": True,
        "lower_SD_thres_for_reasonable_ripple": 1,
        "MAX_EPOCHS": 1,
        "is_debug": False,
        "i_mouse_test": 0,
    }  # same as the first fold

    dlf = DataLoaderFillerFvsT(**DL_CONF)

    dl_tra = dlf.fill("Training")
    dl_test = dlf.fill("Test")

    # ## Checks whether any same data exist on both training and test dataset
    # print(len(dl_tra.dataset.tensors[0]))
    # print(len(dl_test.dataset.tensors[0]))

    # ## Checks the dataleak regarding mouse ID
    # dl_tra.dataset.tensors

    batch = next(iter(dl_tra))
    print(batch[1].unique(return_counts=True))

    # for epoch in range(10):
    #     dl_tra = dlf.fill("Training")
    #     for i_batch, batch in enumerate(dl_tra):
    #         Xb_tra, Tb_tra = batch
    #         print(i_batch)
    #         print(Tb_tra)
