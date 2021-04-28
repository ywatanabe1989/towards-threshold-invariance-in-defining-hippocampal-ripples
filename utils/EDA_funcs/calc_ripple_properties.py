#!/usr/bin/env python
import argparse
import sys; sys.path.append('.')
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

import utils.general as ug
import utils.path_converters as upcvt


def calc_ripple_properties(rip_sec_df, lpath_lfp):
    '''rip_sec_df must have 'start_sec', 'end_sec' columns.
       lpath_lfp is used not only to load LFP
       but also to load MEP magnitude and ripple band magnitude.
    '''
    ################################################################################
    ## Parameters
    ################################################################################
    SAMP_RATE = ug.get_samp_rate_int_from_fpath(lpath_lfp) #1000    

    ################################################################################
    ## PATHs
    ################################################################################
    # lpath_mep_magni_sd_normed = lpath_lfp.replace('orig', 'magni')\
    #                            .replace('_fp16.npy', '_mep_magni_sd_fp16.npy')
    # lpath_ripple_magni_sd_normed = lpath_lfp.replace('orig', 'magni')\
    #                               .replace('_fp16.npy', '_ripple_band_magni_sd_fp16.npy')

    lpath_mep_magni_sd_normed = upcvt.LFP_to_MEP_magni(lpath_lfp)
    lpath_ripple_magni_sd_normed = upcvt.LFP_to_ripple_magni(lpath_lfp)
    ################################################################################
    ## Loads
    ################################################################################
    lfp = np.load(lpath_lfp).squeeze().astype(np.float32).squeeze()
    mep_magni_sd_normed = np.load(lpath_mep_magni_sd_normed).astype(np.float32).squeeze()
    ripple_magni_sd_normed = np.load(lpath_ripple_magni_sd_normed).astype(np.float32).squeeze()

    ################################################################################
    ## Packs sliced LFP, MEP magnitude, and ripple band magnitude into one dataframe
    ################################################################################
    df = rip_sec_df.copy()
    df['duration_ms'] = ((df['end_sec'] - df['start_sec']) * 1000).astype(int)
    df['start_pts'] = (df['start_sec']*SAMP_RATE).astype(int)
    df['end_pts'] = (df['end_sec']*SAMP_RATE).astype(int)
    # df['duration_ms'] = (df['duration']*1000).astype(int)
    # del df['start_sec'], df['end_sec'], df['duration']

    df['LFP'] = [lfp[start_i:end_i]
                 for start_i, end_i in
                 zip (df['start_pts'], df['end_pts'])
                 ]

    df['MEP magni. / SD'] = [mep_magni_sd_normed[start_i:end_i]
                        for start_i, end_i in
                        zip (df['start_pts'], df['end_pts'])
                        ]

    df['ripple magni. / SD'] = [ripple_magni_sd_normed[start_i:end_i]
                           for start_i, end_i in
                           zip (df['start_pts'], df['end_pts'])
                           ]

    ################################################################################
    ## Calculates Properties during ripple candidates
    ################################################################################
    df['ln(duration_ms)'] = np.log(df['duration_ms'])
    df['mean ln(MEP magni. / SD)'] = np.log(df['MEP magni. / SD'].apply(np.mean))
    df['ln(ripple peak magni. / SD)'] = np.log(df['ripple magni. / SD'].apply(np.max))

    # fixme; you might want to exclude unnecesssary columns.
    return df


if __name__ == '__main__':
    lpath_lfp = './data/okada/03/day4/split/LFP_MEP_1kHz_npy/orig/tt6-3_fp16.npy'
    lpath_rip = lpath_lfp.replace('LFP_MEP_1kHz_npy', 'ripple_candi_1kHz_pkl')\
                             .replace('.npy', '.pkl')

    ## Loads
    rip_sec_df = ug.load_pkl(lpath_rip)

    ## Calculates ripple properties
    rip_sec_with_props_df = calc_ripple_properties(rip_sec_df, lpath_lfp)
