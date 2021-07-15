#!/usr/bin/env python

import argparse
from bisect import bisect_left
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
# def mk_rip_peak_magnis_arr(lfps, rips, i_lfp):
#     lfp = lfps[i_lfp]
#     rip = rips[i_lfp]
#     rip_peak_magnis_arr = np.zeros_like(lfp)

#     SAMP_RATE = 1000
#     for i_r, r in rip.iterrows():
#         start_pts = int(r["start_sec"] * SAMP_RATE)
#         end_pts = int(r["end_sec"] * SAMP_RATE)
#         ripple_peak_magni_SD = np.exp(r["ln(ripple peak magni. / SD)"])
#         rip_peak_magnis_arr[start_pts:end_pts] = ripple_peak_magni_SD
#     return rip_peak_magnis_arr


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


import skimage
import utils

i_mouse_test = 0
lfps_tra, rips_tra = utils.pj.load.lfps_rips_tra_or_tes("tra", i_mouse_test)
lfps_tes, rips_tes = utils.pj.load.lfps_rips_tra_or_tes("tes", i_mouse_test)
print(len(lfps_tra), len(lfps_tes))

lfp, rip_sec = lfps_tra[0], rips_tra[0]
kwargs = {
    "samp_rate": 1000,
    'window_size_pts': 400,
    "use_fp16": True,
    "use_shuffle": True,
    "use_random_start": False,
}


def define_Xb_Tb(lfp, rip_sec, **kwargs)
    '''
    Defines LFP epochs.
        Tb:
            'n': not ripple including LFP epoch
            's': just one suspicioius ripple including LFP epoch (1 SD < riple peak magnitude < 7 SD)
            'r': just one reasonable ripple including LFP epoch (7 SD <= riple peak magnitude)        
    '''

    ## Random start
    random_start_pts = (
        random.randint(0, kwargs['window_size_pts']) if kwargs['use_random_start'] else 0
    )

    ## Filled rip_sec_df
    rip_filled = utils.pj.fill_rip_sec(lfp, rip_sec, kwargs['samp_rate'])

    ## Slices LFP
    the_4th_last_rip_end_pts = int(rip_filled.iloc[-4]["end_sec"] * kwargs['samp_rate'])
    _lfp = lfp[random_start_pts:the_4th_last_rip_end_pts] # fixme; _lfp to lfp

    ##############################
    # Xb
    ##############################    
    slices = skimage.util.view_as_windows(
        _lfp, window_shape=(kwargs['window_size_pts'],), step=kwargs['window_size_pts'],
    )
    # time points
    slices_start_pts = (
        np.array([random_start_pts + kwargs['window_size_pts'] * i for i in range(len(slices))]) + 1e-10
    )
    slices_start_sec = slices_start_pts / kwargs['samp_rate']

    # slices_center_pts = slices_start_pts + int(kwargs['window_size_pts'] / 2)
    # slices_center_sec = slices_center_pts / kwargs['samp_rate']

    slices_end_pts = slices_start_pts + kwargs['window_size_pts']
    slices_end_sec = slices_end_pts / kwargs['samp_rate']

    # Links slices and filled ripples; counts ripples from the starting point of each slice
    the_1st_indi = np.array(
        [
            bisect_left(rip_filled["start_sec"].values, slices_start_sec[i]) - 1
            for i in range(len(slices))
        ]
    ) # on the starting points
    the_2nd_indi = the_1st_indi + 1
    the_3rd_indi = the_1st_indi + 2

    the_1st_filled_rips = rip_filled.iloc[the_1st_indi]
    the_2nd_filled_rips = rip_filled.iloc[the_2nd_indi]
    the_3rd_filled_rips = rip_filled.iloc[the_3rd_indi]    

    ##############################
    ## Tb
    ##############################
    ## Base conditions
    are_the_1st_ripple = (the_1st_filled_rips['are_ripple_CNN'] == 1.) 
    are_the_2nd_ripple = (the_2nd_filled_rips['are_ripple_CNN'] == 1.) 
    are_the_3rd_ripple = (the_3rd_filled_rips['are_ripple_CNN'] == 1.)

    are_the_1st_over_the_slice_end = (slices_end_sec < the_1st_filled_rips["end_sec"])
    are_the_2nd_over_the_slice_end = (slices_end_sec < the_2nd_filled_rips["end_sec"])
    are_the_3rd_over_the_slice_end = (slices_end_sec < the_3rd_filled_rips["end_sec"])

    the_2nd_ripple_peak_magnitude_SD = np.exp(the_2nd_filled_rips['ln(ripple peak magni. / SD)'])
    
    # possibly_s_or_r = ~are_the_1st_ripple & ~are_the_1st_over_the_slice_end \
    #                 & are_the_2nd_ripple & ~are_the_2nd_over_the_slice_end \
    #                 & ~are_the_3rd_ripple & are_the_3rd_over_the_slice_end

    possibly_s_or_r = np.vstack([~are_the_1st_ripple,
                                 ~are_the_1st_over_the_slice_end,
                                 are_the_2nd_ripple,
                                 ~are_the_2nd_over_the_slice_end,
                                 ~are_the_3rd_ripple,
                                 are_the_3rd_over_the_slice_end
                                 ]).all(axis=0)
    
    ## Conditions
    # 'n': not ripple including LFP epoch
    are_n = ~are_the_1st_ripple & are_the_1st_over_the_slice_end


    
    # 's': just one suspicioius ripple including LFP epoch (1 SD < riple peak magnitude < 7 SD)
    # are_s = ~are_the_1st_ripple & ~are_the_1st_over_the_slice_end \
    #       & are_the_2nd_ripple & ~are_the_2nd_over_the_slice_end \
    #       & ~are_the_3rd_ripple & are_the_3rd_over_the_slice_end \
    #       & (1 <= the_2nd_ripple_peak_magnitude_SD) \
    #       & (the_2nd_ripple_peak_magnitude_SD < 7)
    are_s = possibly_s_or_r \
          & (1 <= the_2nd_ripple_peak_magnitude_SD) \
          & (the_2nd_ripple_peak_magnitude_SD < 7)
    
    # 'r': just one reasonable ripple including LFP epoch (7 SD <= riple peak magnitude)
    # are_r = ~are_the_1st_ripple & ~are_the_1st_over_the_slice_end \
    #       & are_the_2nd_ripple & ~are_the_2nd_over_the_slice_end \
    #       & ~are_the_3rd_ripple & are_the_3rd_over_the_slice_end
    #       & (7 < the_2nd_ripple_peak_magnitude_SD)
    
    are_r = possibly_s_or_r \
          & (7 <= the_2nd_ripple_peak_magnitude_SD)


    ##############################
    ## Xb
    ##############################
    slices_n = slices[are_n]
    slices_s = slices[are_s]
    slices_r = slices[are_r]    

    Xb = np.vstack([slices_n, slices_s, slices_r])
    Tb = np.array(['n' for _ in range(len(slices_n))]
                + ['s' for _ in range(len(slices_s))]
                + ['r' for _ in range(len(slices_r))] 
                 )

    
    

    # are_the_1st_rips_true_ripple_CNN = ~np.isnan(
    #     rip_filled.iloc[the_1st_indi]["are_ripple_CNN"]
    # )
    # are_the_2nd_rips_true_ripple_CNN = ~np.isnan(
    #     rip_filled.iloc[the_2nd_indi]["are_ripple_CNN"]
    # )
    # are_the_3rd_rips_true_ripple_CNN = ~np.isnan(
    #     rip_filled.iloc[the_3rd_indi]["are_ripple_CNN"]
    # )

    # the_1st_rips_SD = np.exp(rip_filled.iloc[the_1st_indi][
    #     "ln(ripple peak magni. / SD)"
    # ].to_numpy())

    # the_2nd_rips_SD = np.exp(rip_filled.iloc[the_2nd_indi][
    #     "ln(ripple peak magni. / SD)"
    # ].to_numpy())

    # the_3rd_rips_SD = np.exp(rip_filled.iloc[the_3rd_indi][
    #     "ln(ripple peak magni. / SD)"
    # ].to_numpy())

    # # nan to 0
    # the_1st_rips_SD[np.isnan(the_1st_rips_SD)] = 0
    # the_2nd_rips_SD[np.isnan(the_2nd_rips_SD)] = 0
    # the_3rd_rips_SD[np.isnan(the_3rd_rips_SD)] = 0


    # # check if the ripple lasts over the slice's end point
    # are_the_1st_rips_over_the_slice_end = (
    #     (slices_end_sec < rip_filled.iloc[the_1st_indi]["end_sec"])
    #     .to_numpy()
    #     .astype(np.bool)
    # )
    # are_the_2nd_rips_over_the_slice_end = (
    #     (slices_end_sec < rip_filled.iloc[the_2nd_indi]["end_sec"])
    #     .to_numpy()
    #     .astype(np.bool)
    # )
    # are_the_3rd_rips_over_the_slice_end = (
    #     (slices_end_sec < rip_filled.iloc[the_3rd_indi]["end_sec"])
    #     .to_numpy()
    #     .astype(np.bool)
    # )

    

    # ##############################
    # # Tb
    # ##############################    
    # are_noRipple = (the_1st_rips_SD == 0) * are_the_1st_rips_over_the_slice_end

    # ## Choose the samples which completely include only one True_cleaned Ripple
    # include_only_one_Ripple = (
    #     (the_1st_rips_SD == 0)
    #     & ~are_the_1st_rips_over_the_slice_end
    #     & (the_2nd_rips_SD > 1)
    #     & ~are_the_2nd_rips_over_the_slice_end
    #     & are_the_2nd_rips_true_ripple_CNN
    #     & (the_3rd_rips_SD == 0)
    #     & are_the_3rd_rips_over_the_slice_end
    # )

    # Tb_SD = np.nan * np.ones(len(slices), dtype=np.float) # initialize
    # Tb_SD[are_noRipple] = 0
    # Tb_SD[include_only_one_Ripple] = the_2nd_rips_SD[include_only_one_Ripple]


    # # Excludes undefined slices
    # are_defined = Tb_SD != -1
    # Xb, Tb_SD = slices[are_defined], Tb_SD[are_defined]


    # # Excludes Samples including a 1 <= SD < 7 Ripple
    # min_SD = 7
    # are_noRipple = Tb_SD == 0
    # are_Ripple = min_SD <= Tb_SD
    # indi = are_noRipple + are_Ripple
    # Xb, Tb_SD = Xb[indi], Tb_SD[indi]
    # # Convert SD to Label to Binary Classification
    # Tb = Tb_SD
    # Tb[min_SD <= Tb] = 1
    # Tb = Tb.astype(np.int)
    # assert len(Xb) == len(Tb)
    # return {"Xb": Xb, "Tb": Tb}
