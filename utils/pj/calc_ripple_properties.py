#!/usr/bin/env python
import argparse
import sys

sys.path.append(".")
import os

import numpy as np
import pandas as pd
import utils
from tqdm import tqdm

# import utils.general as ug
# import utils.path_converters as upcvt


def calc_ripple_properties(lpath_lfp):
    ################################################################################
    ## Parameters
    ################################################################################
    SAMP_RATE = utils.pj.get_samp_rate_int_from_fpath(lpath_lfp)  # 1000

    ################################################################################
    ## PATHs
    ################################################################################
    lpath_rip = utils.pj.path_converters.LFP_to_ripples(
        lpath_lfp, rip_sec_ver="candi_orig"
    )
    lpath_mep_magni_sd_normed = utils.pj.path_converters.LFP_to_MEP_magni(lpath_lfp)
    lpath_ripple_magni_sd_normed = utils.pj.path_converters.LFP_to_ripple_magni(
        lpath_lfp
    )

    ################################################################################
    ## Loads
    ################################################################################
    lfp = utils.general.load(lpath_lfp).astype(np.float32).squeeze()
    rip_sec_df = utils.general.load(lpath_rip)
    mep_magni_sd_normed = (
        utils.general.load(lpath_mep_magni_sd_normed).astype(np.float32).squeeze()
    )
    ripple_magni_sd_normed = (
        utils.general.load(lpath_ripple_magni_sd_normed).astype(np.float32).squeeze()
    )

    ################################################################################
    ## Packs sliced LFP, MEP magnitude, and ripple band magnitude into one dataframe
    ################################################################################
    df = rip_sec_df.copy()
    df["duration_ms"] = ((df["end_sec"] - df["start_sec"]) * 1000).astype(int)
    df["start_pts"] = (df["start_sec"] * SAMP_RATE).astype(int)
    df["end_pts"] = (df["end_sec"] * SAMP_RATE).astype(int)
    # df['duration_ms'] = (df['duration']*1000).astype(int)
    # del df['start_sec'], df['end_sec'], df['duration']

    df["LFP"] = [
        lfp[start_i:end_i] for start_i, end_i in zip(df["start_pts"], df["end_pts"])
    ]

    df["MEP magni. / SD"] = [
        mep_magni_sd_normed[start_i:end_i]
        for start_i, end_i in zip(df["start_pts"], df["end_pts"])
    ]

    df["ripple magni. / SD"] = [
        ripple_magni_sd_normed[start_i:end_i]
        for start_i, end_i in zip(df["start_pts"], df["end_pts"])
    ]

    ################################################################################
    ## Calculates Properties during ripple candidates
    ################################################################################
    df["ln(duration_ms)"] = np.log(df["duration_ms"])
    df["ln(mean MEP magni. / SD)"] = np.log(df["MEP magni. / SD"].apply(np.mean))
    df["ln(ripple peak magni. / SD)"] = np.log(df["ripple magni. / SD"].apply(np.max))

    # fixme; you might want to exclude unnecesssary columns.

    keys_to_delete = [
        "duration_ms",
        "start_pts",
        "end_pts",
        "LFP",
        "MEP magni. / SD",
        "ripple magni. / SD",
    ]
    for k in keys_to_delete:
        del df[k]
    return df


if __name__ == "__main__":
    lpath_lfp = "./data/okada/03/day4/split/LFP_MEP_1kHz_npy/orig/tt6-3_fp16.npy"

    # lpath_rip = lpath_lfp.replace('LFP_MEP_1kHz_npy', 'ripple_candi_1kHz_pkl')\
    #                          .replace('.npy', '.pkl')

    # ## Loads
    # rip_sec_df = utils.general.load_pkl(lpath_rip)

    ## Calculates ripple properties
    # rip_sec_with_props_df = calc_ripple_properties(rip_sec_df, lpath_lfp)
    rip_sec_with_props_df = calc_ripple_properties(lpath_lfp)
