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


import skimage
import utils

i_mouse_test = 0
lfps_tra, rips_tra = utils.pj.load.lfps_rips_tra_or_tes("tra", i_mouse_test)
lfps_tes, rips_tes = utils.pj.load.lfps_rips_tra_or_tes("tes", i_mouse_test)
print(len(lfps_tra), len(lfps_tes))

lfp, rip_sec = lfps_tra[0], rips_tra[0]
kwargs = {
    "samp_rate": 1000,
    "use_fp16": True,
    "use_shuffle": True,
    "max_seq_len_pts": 200,
    "step": None,
    "use_perturb": False,
    "max_distancems": None,
    "ripples_binary_classification": False,
    "ripples_multi_classification": True,
    "estimates_ripple_params": False,
    "label_name": None,
}

## koko
from bisect import bisect_left


# def define_Xb_Tb(lfp, rip_sec, **kwargs):
#     """
#     n_mouse_tes = '02'
#     kwargs = {'samp_rate':1000,
#               'use_fp16':True,
#               'use_shuffle':True,
#               'max_seq_len_pts':200,
#               'step':None,
#               'use_perturb':False,
#               'max_distancems':None,
#               'ripples_binary_classification':False,
#               'ripples_multi_classification':True,
#               'estimates_ripple_params':False,
#               'label_name':None,
#               }
#     #          'label_name':None,
#     lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tes'], **kwargs)
#     lfp, rip_sec = lfps[0], rips_sec[0]

#     outputs = define_Xb_Tb(lfp, rip_sec, **kwargs)
#     print(np.unique(outputs['Tb_label']))
#     print(outputs.keys())
#     """
#     samp_rate = kwargs.get("samp_rate", 1000)
#     max_seq_len_pts = kwargs.get("max_seq_len_pts", 400)
#     # max_distance_ms = kwargs.get('max_distance_ms') if kwargs.get('max_distance_ms') else int(max_seq_len_pts/2)
#     dtype = np.float16 if kwargs.get("use_fp16", True) else np.float32

#     perturb_pts = (
#         np.random.randint(0, max_seq_len_pts) if kwargs.get("use_perturb", False) else 0
#     )
#     perturb_sec = perturb_pts / samp_rate

#     rip_sec_cp = rip_sec.copy()

#     the_4th_to_last_rip_end_pts = int(rip_sec_cp.iloc[-4]["end_sec"] * samp_rate)
#     lfp = lfp[perturb_pts:the_4th_to_last_rip_end_pts]

#     step = kwargs.get("step") if kwargs.get("step") else max_seq_len_pts

#     # Xb
#     slices = skimage.util.view_as_windows(
#         lfp, window_shape=(max_seq_len_pts,), step=step
#     )

#     slices_start_pts = (
#         np.array([perturb_pts + step * i for i in range(len(slices))]) + 1e-10
#     )
#     slices_start_sec = slices_start_pts / samp_rate

#     slices_center_pts = slices_start_pts + int(max_seq_len_pts / 2)
#     slices_center_sec = slices_center_pts / samp_rate

#     slices_end_pts = slices_start_pts + max_seq_len_pts
#     slices_end_sec = slices_end_pts / samp_rate

#     the_1st_rips_indi = np.array(
#         [
#             bisect_left(rip_sec_cp["start_sec"].values, slices_start_sec[i]) - 1
#             for i in range(len(slices))
#         ]
#     )
#     the_2nd_rips_indi = the_1st_rips_indi + 1
#     the_3rd_rips_indi = the_1st_rips_indi + 2


#     label_name = kwargs["label_name"]
#     if label_name is None:
#         are_the_1st_rips_true_ripple = ~np.isnan(
#             rip_sec_cp.iloc[the_1st_rips_indi]["are_ripple_CNN"]
#         )
#         are_the_2nd_rips_true_ripple = ~np.isnan(
#             rip_sec_cp.iloc[the_2nd_rips_indi]["are_ripple_CNN"]
#         )
#         are_the_3rd_rips_true_ripple = ~np.isnan(
#             rip_sec_cp.iloc[the_3rd_rips_indi]["are_ripple_CNN"]
#         )
#     if label_name is not None:
#         cls_id = {
#             "padding": -10,
#             "back_ground": -2,
#             "true_ripple": 0,
#             "false_ripple": 1,
#         }

#         the_1st_rips_label_cleaned = rip_sec_cp.iloc[the_1st_rips_indi][
#             label_name
#         ].to_numpy()
#         the_2nd_rips_label_cleaned = rip_sec_cp.iloc[the_2nd_rips_indi][
#             label_name
#         ].to_numpy()
#         the_3rd_rips_label_cleaned = rip_sec_cp.iloc[the_3rd_rips_indi][
#             label_name
#         ].to_numpy()

#         are_the_1st_rips_true_ripple = (
#             the_1st_rips_label_cleaned == cls_id["true_ripple"]
#         )
#         are_the_2nd_rips_true_ripple = (
#             the_2nd_rips_label_cleaned == cls_id["true_ripple"]
#         )
#         are_the_3rd_rips_true_ripple = (
#             the_3rd_rips_label_cleaned == cls_id["true_ripple"]
#         )

#     the_1st_rips_SD = rip_sec_cp.iloc[the_1st_rips_indi][
#         "ln(ripple peak magni. / SD)"
#     ].to_numpy()
#     the_2nd_rips_SD = rip_sec_cp.iloc[the_2nd_rips_indi][
#         "ln(ripple peak magni. / SD)"
#     ].to_numpy()
#     the_3rd_rips_SD = rip_sec_cp.iloc[the_3rd_rips_indi][
#         "ln(ripple peak magni. / SD)"
#     ].to_numpy()

#     the_1st_rips_SD[np.isnan(the_1st_rips_SD)] = 0
#     the_2nd_rips_SD[np.isnan(the_2nd_rips_SD)] = 0
#     the_3rd_rips_SD[np.isnan(the_3rd_rips_SD)] = 0

#     the_1st_rips_SD = np.exp(the_1st_rips_SD)
#     the_2nd_rips_SD = np.exp(the_2nd_rips_SD)
#     the_3rd_rips_SD = np.exp(the_3rd_rips_SD)

#     are_the_1st_rips_over_the_slice_end = (
#         (slices_end_sec < rip_sec_cp.iloc[the_1st_rips_indi]["end_sec"])
#         .to_numpy()
#         .astype(np.bool)
#     )
#     are_the_2nd_rips_over_the_slice_end = (
#         (slices_end_sec < rip_sec_cp.iloc[the_2nd_rips_indi]["end_sec"])
#         .to_numpy()
#         .astype(np.bool)
#     )
#     are_the_3rd_rips_over_the_slice_end = (
#         (slices_end_sec < rip_sec_cp.iloc[the_3rd_rips_indi]["end_sec"])
#         .to_numpy()
#         .astype(np.bool)
#     )

#     # Tb_label
#     are_noRipple = (the_1st_rips_SD == 0) * are_the_1st_rips_over_the_slice_end

#     ## Choose the sample which include only one True_cleaned Ripple
#     include_only_one_Ripple = (
#         (the_1st_rips_SD == 0)
#         & ~are_the_1st_rips_over_the_slice_end
#         & (the_2nd_rips_SD > 1)
#         & ~are_the_2nd_rips_over_the_slice_end
#         & are_the_2nd_rips_true_ripple
#         & (the_3rd_rips_SD == 0)
#         & are_the_3rd_rips_over_the_slice_end
#     )

#     Tb_SD = np.ones(len(slices), dtype=np.float) * (-1)  # initialize
#     Tb_SD[are_noRipple] = 0
#     Tb_SD[include_only_one_Ripple] = the_2nd_rips_SD[include_only_one_Ripple]

#     """
#   ## Confirm Tb_label
#   # 1) print numbers
#   i = np.random.randint(1e5)
#   print('slices start: {}'.format(slices_start_sec[i:i+9]))
#   print('slices end  : {}'.format(slices_end_sec[i:i+9]))
#   print('slices label: {}'.format(Tb_label[i:i+9]))
#   print(rip_sec[(slices_start_sec[i] < rip_sec['end_sec']) & (rip_sec['start_sec'] < slices_end_sec[i+9])])

#   # 2) plot # fixme
#   i = np.random.randint(1e2)
#   n_slices = 10
#   _start_sec, _end_sec = slices_start_sec[i], slices_end_sec[i+n_slices]
#   _start_pts, _end_pts = int(_start_sec*samp_rate), int(_end_sec*samp_rate)
#   _lfp = lfp[_start_pts:_end_pts]
#   _rip_sec = rip_sec_cp[(_start_sec < rip_sec_cp['start_sec']) & (rip_sec_cp['end_sec'] < _end_sec)] # fixme
#   _slices = slices[i:i+n_slices]
#   _slices_start_sec = slices_start_sec[i:i+n_slices]
#   _slices_end_sec = slices_end_sec[i:i+n_slices]

#   t = 1.*np.arange(_start_sec, _end_sec, 1./samp_rate)
#   plt.plot(t, _lfp)

#   for rip in _rip_sec.itertuples():
#     # ax.axvspan(rip.start_sec, ax.end_sec, alpha=0.3, color='red')
#     t_rip = 1.*np.arange(rip.start_sec, rip.end_sec, 1./samp_rate)
#     plt.plot(t_rip, np.ones_like(t_rip)*rip.level*100, color='red')

#   for i_slice in range(n_slices):
#     t_slice = 1.*np.arange(_slices_start_sec[i_slice], _slices_end_sec[i_slice], 1./samp_rate)
#     plt.plot(t_slice, np.ones_like(t_slice)*Tb_label[i_slice].numpy()*100, color='green')
#   """

#     assert (
#         kwargs["ripples_binary_classification"]
#         or kwargs["ripples_multi_classification"]
#         or kwargs["estimates_ripple_params"]
#     )

#     if (
#         kwargs["ripples_binary_classification"]
#         or kwargs["ripples_multi_classification"]
#     ):  ## IMPORTANT
#         # Excludes undefined slices
#         are_defined = Tb_SD != -1
#         Xb, Tb_SD = slices[are_defined], Tb_SD[are_defined]

#         if kwargs["ripples_multi_classification"]:
#             outputs = {"Xb": Xb, "Tb_SD": Tb_SD}

#         if kwargs["ripples_binary_classification"]:
#             min_SD = 5
#             # Excludes Samples including a 1 <= SD < 5 Ripple
#             are_noRipple = Tb_SD == 0
#             are_Ripple = min_SD <= Tb_SD
#             indi = are_noRipple + are_Ripple
#             Xb, Tb_SD = Xb[indi], Tb_SD[indi]
#             # Convert SD to Label to Binary Classification
#             Tb_label = Tb_SD
#             Tb_label[min_SD <= Tb_label] = 1
#             Tb_label = Tb_label.astype(np.int)
#             assert len(Xb) == len(Tb_label)
#             outputs = {"Xb": Xb, "Tb_label": Tb_label}

#     """
#   # Check the outputs' lengthes
#   keys = outputs.keys()
#   first = True
#   for k in keys:
#     if first:
#       length = len(outputs[k])
#       first = False
#     if not first:
#       length_next = len(outputs[k])
#       assert length_next == length
#       length_next = length
#   """

#     # if kwargs["use_shuffle"]:  # 2nd shuffle
#     #     outputs = mf.shuffle_dict(outputs)

#     return outputs




################################################################################
## def define_Xb_Tb(lfp, rip_sec, **kwargs)


################################################################################
## parameters
################################################################################
window_size_pts = 400
use_random_start = True
samp_rate = 1000
################################################################################

def fill_undefined_rip_sec(lfp, rip_sec, samp_rate):
    rip_sec_level_0 = pd.DataFrame()
    rip_sec_level_0['end_sec'] = rip_sec['start_sec'] - 1/samp_rate
    rip_sec_level_0['start_sec'] = np.hstack((np.array(0) - 1/samp_rate, rip_sec['end_sec'][:-1].values)) + 1/samp_rate
    cols = rip_sec_level_0.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    rip_sec_level_0 = rip_sec_level_0[cols]
    # add last row
    lfp_end_sec = len(lfp) / samp_rate
    last_row_dict = {'start_sec':[rip_sec.iloc[-1]['end_sec']],
                     'end_sec':[lfp_end_sec],
                     }
    last_row_index = rip_sec_level_0.index[-1:] + 1
    last_row_df = pd.DataFrame(data=last_row_dict).set_index(last_row_index)
    rip_sec_level_0 = rip_sec_level_0.append(last_row_df)

    keys = rip_sec.keys()
    filled_ripples = pd.concat([rip_sec,
                                rip_sec_level_0],
                                sort=False)[keys]
    filled_ripples = filled_ripples.sort_values('start_sec')
    
    return filled_ripples


## Fill undefined times as False Ripple time


# random_start_pts = (
#     random.randint(0, window_size_pts) if use_random_start else 0
# )
random_start_pts = 0

rip_sec_cp = fill_undefined_rip_sec(lfp, rip_sec, samp_rate)
# rip_sec_cp = rip_sec.copy()

the_4th_last_rip_end_pts = int(rip_sec_cp.iloc[-4]["end_sec"] * samp_rate)
lfp = lfp[random_start_pts:the_4th_last_rip_end_pts]

# Xb
slices = skimage.util.view_as_windows(
    lfp, window_shape=(window_size_pts,), step=window_size_pts,
)
# time points of Xb
slices_start_pts = (
    np.array([random_start_pts + window_size_pts * i for i in range(len(slices))]) + 1e-10
)
slices_start_sec = slices_start_pts / samp_rate

slices_center_pts = slices_start_pts + int(window_size_pts / 2)
slices_center_sec = slices_center_pts / samp_rate

slices_end_pts = slices_start_pts + window_size_pts
slices_end_sec = slices_end_pts / samp_rate

# finds the indice of close ripples
# the_1st_rips_indi = np.array(
#     [
#         bisect_left(rip_sec_cp["start_sec"].values, slices_start_sec[i])
#         for i in range(len(slices))
#     ]
# )



the_1st_rips_indi = np.array(
    [
        bisect_left(rip_sec_cp["start_sec"].values, slices_start_sec[i]) - 1
        for i in range(len(slices))
    ]
)
the_2nd_rips_indi = the_1st_rips_indi + 1
the_3rd_rips_indi = the_1st_rips_indi + 2

# are_the_1st_rips_true_ripple = \
#     rip_sec_cp.iloc[the_1st_rips_indi]["are_ripple_CNN"]

# are_the_2nd_rips_true_ripple = \
#     rip_sec_cp.iloc[the_2nd_rips_indi]["are_ripple_CNN"]

# are_the_3rd_rips_true_ripple = \
#     rip_sec_cp.iloc[the_3rd_rips_indi]["are_ripple_CNN"]


are_the_1st_rips_true_ripple = ~np.isnan(
    rip_sec_cp.iloc[the_1st_rips_indi]["are_ripple_CNN"]
)
are_the_2nd_rips_true_ripple = ~np.isnan(
    rip_sec_cp.iloc[the_2nd_rips_indi]["are_ripple_CNN"]
)
are_the_3rd_rips_true_ripple = ~np.isnan(
    rip_sec_cp.iloc[the_3rd_rips_indi]["are_ripple_CNN"]
)

    # # label_name = kwargs["label_name"]
    # if label_name is None:
    #     are_the_1st_rips_true_ripple = ~np.isnan(
    #         rip_sec_cp.iloc[the_1st_rips_indi]["are_ripple_CNN"]
    #     )
    #     are_the_2nd_rips_true_ripple = ~np.isnan(
    #         rip_sec_cp.iloc[the_2nd_rips_indi]["are_ripple_CNN"]
    #     )
    #     are_the_3rd_rips_true_ripple = ~np.isnan(
    #         rip_sec_cp.iloc[the_3rd_rips_indi]["are_ripple_CNN"]
    #     )
    # if label_name is not None:
    #     cls_id = {
    #         "padding": -10,
    #         "back_ground": -2,
    #         "true_ripple": 0,
    #         "false_ripple": 1,
    #     }

    #     the_1st_rips_label_cleaned = rip_sec_cp.iloc[the_1st_rips_indi][
    #         label_name
    #     ].to_numpy()
    #     the_2nd_rips_label_cleaned = rip_sec_cp.iloc[the_2nd_rips_indi][
    #         label_name
    #     ].to_numpy()
    #     the_3rd_rips_label_cleaned = rip_sec_cp.iloc[the_3rd_rips_indi][
    #         label_name
    #     ].to_numpy()

    #     are_the_1st_rips_true_ripple = (
    #         the_1st_rips_label_cleaned == cls_id["true_ripple"]
    #     )
    #     are_the_2nd_rips_true_ripple = (
    #         the_2nd_rips_label_cleaned == cls_id["true_ripple"]
    #     )
    #     are_the_3rd_rips_true_ripple = (
    #         the_3rd_rips_label_cleaned == cls_id["true_ripple"]
    #     )


the_1st_rips_SD = np.exp(rip_sec_cp.iloc[the_1st_rips_indi][
    "ln(ripple peak magni. / SD)"
].to_numpy())

the_2nd_rips_SD = np.exp(rip_sec_cp.iloc[the_2nd_rips_indi][
    "ln(ripple peak magni. / SD)"
].to_numpy())

the_3rd_rips_SD = np.exp(rip_sec_cp.iloc[the_3rd_rips_indi][
    "ln(ripple peak magni. / SD)"
].to_numpy())

the_1st_rips_SD[np.isnan(the_1st_rips_SD)] = 0
the_2nd_rips_SD[np.isnan(the_2nd_rips_SD)] = 0
the_3rd_rips_SD[np.isnan(the_3rd_rips_SD)] = 0

    # the_1st_rips_SD = np.exp(the_1st_rips_SD)
    # the_2nd_rips_SD = np.exp(the_2nd_rips_SD)
    # the_3rd_rips_SD = np.exp(the_3rd_rips_SD)

# are_the_1st_rip_included = \
# (slices_start_sec < rip_sec_cp.iloc[the_1st_rips_indi]['start_sec']) \
# & \
# (rip_sec_cp.iloc[the_1st_rips_indi]['end_sec'] < slices_end_sec )

    
are_the_1st_rips_over_the_slice_end = (
    (slices_end_sec < rip_sec_cp.iloc[the_1st_rips_indi]["end_sec"])
    .to_numpy()
    .astype(np.bool)
)
are_the_2nd_rips_over_the_slice_end = (
    (slices_end_sec < rip_sec_cp.iloc[the_2nd_rips_indi]["end_sec"])
    .to_numpy()
    .astype(np.bool)
)
are_the_3rd_rips_over_the_slice_end = (
    (slices_end_sec < rip_sec_cp.iloc[the_3rd_rips_indi]["end_sec"])
    .to_numpy()
    .astype(np.bool)
)

# Tb_label
are_noRipple = (the_1st_rips_SD == 0) * are_the_1st_rips_over_the_slice_end

## Choose the sample which include only one True_cleaned Ripple
include_only_one_Ripple = (
    (the_1st_rips_SD == 0)
    & ~are_the_1st_rips_over_the_slice_end
    & (the_2nd_rips_SD > 1)
    & ~are_the_2nd_rips_over_the_slice_end
    & are_the_2nd_rips_true_ripple
    & (the_3rd_rips_SD == 0)
    & are_the_3rd_rips_over_the_slice_end
)

Tb_SD = np.ones(len(slices), dtype=np.float) * (-1)  # initialize
Tb_SD[are_noRipple] = 0
Tb_SD[include_only_one_Ripple] = the_2nd_rips_SD[include_only_one_Ripple]

  #   """
  # ## Confirm Tb_label
  # # 1) print numbers
  # i = np.random.randint(1e5)
  # print('slices start: {}'.format(slices_start_sec[i:i+9]))
  # print('slices end  : {}'.format(slices_end_sec[i:i+9]))
  # print('slices label: {}'.format(Tb_label[i:i+9]))
  # print(rip_sec[(slices_start_sec[i] < rip_sec['end_sec']) & (rip_sec['start_sec'] < slices_end_sec[i+9])])

  # # 2) plot # fixme
  # i = np.random.randint(1e2)
  # n_slices = 10
  # _start_sec, _end_sec = slices_start_sec[i], slices_end_sec[i+n_slices]
  # _start_pts, _end_pts = int(_start_sec*samp_rate), int(_end_sec*samp_rate)
  # _lfp = lfp[_start_pts:_end_pts]
  # _rip_sec = rip_sec_cp[(_start_sec < rip_sec_cp['start_sec']) & (rip_sec_cp['end_sec'] < _end_sec)] # fixme
  # _slices = slices[i:i+n_slices]
  # _slices_start_sec = slices_start_sec[i:i+n_slices]
  # _slices_end_sec = slices_end_sec[i:i+n_slices]

  # t = 1.*np.arange(_start_sec, _end_sec, 1./samp_rate)
  # plt.plot(t, _lfp)

  # for rip in _rip_sec.itertuples():
  #   # ax.axvspan(rip.start_sec, ax.end_sec, alpha=0.3, color='red')
  #   t_rip = 1.*np.arange(rip.start_sec, rip.end_sec, 1./samp_rate)
  #   plt.plot(t_rip, np.ones_like(t_rip)*rip.level*100, color='red')

  # for i_slice in range(n_slices):
  #   t_slice = 1.*np.arange(_slices_start_sec[i_slice], _slices_end_sec[i_slice], 1./samp_rate)
  #   plt.plot(t_slice, np.ones_like(t_slice)*Tb_label[i_slice].numpy()*100, color='green')
  # """

  #   assert (
  #       kwargs["ripples_binary_classification"]
  #       or kwargs["ripples_multi_classification"]
  #       or kwargs["estimates_ripple_params"]
  #   )

  #   if (
  #       kwargs["ripples_binary_classification"]
  #       or kwargs["ripples_multi_classification"]
  #   ):  ## IMPORTANT
  #       # Excludes undefined slices
  #       are_defined = Tb_SD != -1
  #       Xb, Tb_SD = slices[are_defined], Tb_SD[are_defined]

  #       if kwargs["ripples_multi_classification"]:
  #           outputs = {"Xb": Xb, "Tb_SD": Tb_SD}

  #       if kwargs["ripples_binary_classification"]:
  #           min_SD = 5
  #           # Excludes Samples including a 1 <= SD < 5 Ripple
  #           are_noRipple = Tb_SD == 0
  #           are_Ripple = min_SD <= Tb_SD
  #           indi = are_noRipple + are_Ripple
  #           Xb, Tb_SD = Xb[indi], Tb_SD[indi]
  #           # Convert SD to Label to Binary Classification
  #           Tb_label = Tb_SD
  #           Tb_label[min_SD <= Tb_label] = 1
  #           Tb_label = Tb_label.astype(np.int)
  #           assert len(Xb) == len(Tb_label)
  #           outputs = {"Xb": Xb, "Tb_label": Tb_label}

# Excludes undefined slices
are_defined = Tb_SD != -1
Xb, Tb_SD = slices[are_defined], Tb_SD[are_defined]


# Excludes Samples including a 1 <= SD < 7 Ripple
min_SD = 7
are_noRipple = Tb_SD == 0
are_Ripple = min_SD <= Tb_SD
indi = are_noRipple + are_Ripple
Xb, Tb_SD = Xb[indi], Tb_SD[indi]
# Convert SD to Label to Binary Classification
Tb_label = Tb_SD
Tb_label[min_SD <= Tb_label] = 1
Tb_label = Tb_label.astype(np.int)
assert len(Xb) == len(Tb_label)
outputs = {"Xb": Xb, "Tb_label": Tb_label}

            
    """
  # Check the outputs' lengthes
  keys = outputs.keys()
  first = True
  for k in keys:
    if first:
      length = len(outputs[k])
      first = False
    if not first:
      length_next = len(outputs[k])
      assert length_next == length
      length_next = length
  """

    # if kwargs["use_shuffle"]:  # 2nd shuffle
    #     outputs = mf.shuffle_dict(outputs)
