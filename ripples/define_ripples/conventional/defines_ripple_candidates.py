#!/usr/bin/env python

import numpy as np
import os
import pandas as pd

import sys; sys.path.append('.')
from modules.rippledetection.core import (exclude_close_events,
                                          filter_band,
                                          gaussian_smooth,
                                          threshold_by_zscore)
                           



## Functions
def define_ripple_candidates(time_x, lfp, samp_rate,
                             lo_hz=100,
                             hi_hz=250,
                             minimum_duration=0.015,
                             zscore_threshold=1.0,
                             smoothing_sigma=0.004,
                             close_ripple_threshold=0.0):

    not_null = np.all(pd.notnull(lfp), axis=1)

    lfp, time_x = lfp[not_null], time_x[not_null]

    filtered_lfps = np.stack(
        [filter_band(lfp, samp_rate, lo_hz=lo_hz, hi_hz=hi_hz) for lfp in lfp.T])

    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=0)

    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, samp_rate)

    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)
    filted_magni = combined_filtered_lfps # aliase    

    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time_x, minimum_duration, zscore_threshold)

    ripple_times = exclude_close_events(
        candidate_ripple_times, close_ripple_threshold)

    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')

    rip_sec = pd.DataFrame(ripple_times, columns=['start_sec', 'end_sec'],
                        index=index)

    return filtered_lfps, filted_magni, rip_sec


if __name__ == '__main__':
    import argparse
    import utils.general as ug
    
    ts = ug.TimeStamper()

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-n", "--npy_fpath",
                    default='./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy', \
                    help="The path of the input lfp file (.npy)")
    args = ap.parse_args()
    
    ## Parameters
    samp_rate = ug.to_int_samp_rate(ug.get_samp_rate_str_from_fpath(args.npy_fpath))


    ## Loads
    fpath = args.npy_fpath
    lfp = np.load(fpath).squeeze().astype(np.float32)
    lfp = lfp[:, np.newaxis]

    start_sec, end_sec, step_sec = 0, 1.*len(lfp)/samp_rate, 1.0/samp_rate
    time_x = np.arange(start_sec, end_sec, step_sec)
    # lfp = lfp[int(start_sec*samp_rate):int(end_sec*samp_rate)]


    ## Detects Ripple Candidates
    print('Detecting ripples from {} (Length: {:.1f}h)'.format(fpath, len(lfp)/samp_rate/3600))
    lo_hz_ripple, hi_hz_ripple = ug.load_yaml_as_dict('./conf/global.yaml')['RIPPLE_CANDI_LIM_HZ']
    # 150 Hz, 250 Hz
    _, _, rip_sec = detect_ripple_candidates(time_x,
                                             lfp,
                                             samp_rate,
                                             lo_hz=lo_hz_ripple,
                                             hi_hz=hi_hz_ripple,
                                             zscore_threshold=1)
    ts('')


    ## Renames columns
    rip_sec['start_sec'] = rip_sec['start_time']
    rip_sec['end_sec'] = rip_sec['end_time']
    del rip_sec['start_time'], rip_sec['end_time'], rip_sec['duration']


    ## Save the ripple candidates
    ldir, fname, ext = ug.split_fpath(fpath)
    sdir = ldir.replace('LFP_MEP', 'ripples')\
               .replace('/orig/', '/candi_orig/')\
               .replace('npy', 'pkl')
    spath = sdir + fname + '.pkl'
    ug.save(rip_sec, spath)
    # Saved to: './data/okada/01/day1/split/ripples_1kHz_pkl/candi_orig/tt2-1_fp16.pkl'

    ## EOF
