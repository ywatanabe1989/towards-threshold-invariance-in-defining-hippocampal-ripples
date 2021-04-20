#!/usr/bin/env python
import argparse
import sys; sys.path.append('.')
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from utils.general import (load_pkl,
                           save_pkl,
                           split_fpath,
                           get_samp_rate_str_from_fpath,
                           to_int_samp_rate,
                           )


def calc_ripple_properties(ripples_sec_df, lpath_lfp):
    '''ripples_df must have 'start_time', 'end_time', and 'duration' columns in the unit of seconds.
       lpath_lfp is used not only to load LFP but also to load MEP magnitude and ripple band magnitude.
    '''
    ################################################################################
    ## Parameters
    ################################################################################
    SAMP_RATE = to_int_samp_rate(get_samp_rate_str_from_fpath(args.npy_fpath)) #1000

    ################################################################################
    ## PATHs
    ################################################################################
    lpath_mep_magni = lpath_lfp.replace('orig', 'magni')\
                               .replace('_fp16.npy', '_mep_magni_sd_fp16.npy')
    lpath_ripple_magni = lpath_lfp.replace('orig', 'magni')\
                                  .replace('_fp16.npy', '_ripple_band_magni_sd_fp16.npy')

    ################################################################################
    ## Loads
    ################################################################################
    lfp = np.load(lpath_lfp).squeeze().astype(np.float32).squeeze()
    mep_magni = np.load(lpath_mep_magni).astype(np.float32).squeeze()
    ripple_magni = np.load(lpath_ripple_magni).astype(np.float32).squeeze()

    ################################################################################
    ## Packs sliced LFP, MEP magnitude, and ripple band magnitude into one dataframe
    ################################################################################
    df = ripples_sec_df.copy()
    df['start_pts'] = (df['start_time']*SAMP_RATE).astype(int)
    df['end_pts'] = (df['end_time']*SAMP_RATE).astype(int)
    df['duration_ms'] = (df['duration']*1000).astype(int)
    # del df['start_time'], df['end_time'], df['duration']

    df['LFP'] = [lfp[start_i:end_i]
                 for start_i, end_i in
                 zip (df['start_pts'], df['end_pts'])
                 ]

    df['MEP magni.'] = [mep_magni[start_i:end_i]
                        for start_i, end_i in
                        zip (df['start_pts'], df['end_pts'])
                        ]

    df['ripple magni.'] = [ripple_magni[start_i:end_i]
                           for start_i, end_i in
                           zip (df['start_pts'], df['end_pts'])
                           ]

    ################################################################################
    ## Calculates Properties during ripple candidates
    ################################################################################
    df['ln(duration_ms)'] = np.log(df['duration_ms'])
    df['mean ln(MEP magni.)'] = np.log(df['MEP magni.'].apply(np.mean))
    df['ln(ripple peak magni.)'] = np.log(df['ripple magni.'].apply(np.max))
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-n", "--npy_fpath",
                    default='./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy', \
                    help="The path of the input lfp file (.npy)")
    args = ap.parse_args()

    ## PATHs
    lpath_lfp = args.npy_fpath
    lpath_ripples = lpath_lfp.replace('LFP_MEP_1kHz_npy', 'ripple_candi_1kHz_pkl')\
                             .replace('.npy', '.pkl')

    ## Loads
    ripples_sec_df = load_pkl(lpath_ripples)[['start_time', 'end_time', 'duration']]

    ## Calculates ripple properties
    ripples_sec_df_with_props = calc_ripple_properties(ripples_sec_df, lpath_lfp)

    ## Saves
    spath = lpath_ripples.replace('orig', 'with_props')
    sdir, _, _ = split_fpath(spath)
    os.makedirs(sdir, exist_ok=True)
    save_pkl(ripples_sec_df_with_props, spath)


    ## EOF
