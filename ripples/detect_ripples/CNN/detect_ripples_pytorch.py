import argparse
import numpy as np
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
sys.path.append('05_Ripple/rippledetection/')
from core import (exclude_close_events, exclude_movement, filter_ripple_band,
                   gaussian_smooth, get_envelope,
                   get_multiunit_population_firing_rate,
                   merge_overlapping_ranges, threshold_by_zscore)
import pandas as pd
import myutils.myfunc as mf

mytime = mf.time_tracker()

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath", default='../data/01/day1/split/1kHz/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()

## Funcs
def detect_ripple_candi(time_x, lfp, samp_rate,
                        lo_hz=150,
                        hi_hz=250,
                        minimum_duration=0.015,
                        zscore_threshold=1.0,
                        smoothing_sigma=0.004,
                        close_ripple_threshold=0.0):
    '''
    Baseline by Kay et al, 2016.
    '''

    not_null = np.all(pd.notnull(lfp), axis=1)

    lfp, time_x = lfp[not_null], time_x[not_null]

    filtered_lfps = np.stack(
        [filter_ripple_band(lfp, samp_rate, lo_hz=100, hi_hz=250) for lfp in lfp.T])

    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=0)

    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, samp_rate)

    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)

    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time_x, minimum_duration, zscore_threshold)

    ripple_times = exclude_close_events(
        candidate_ripple_times, close_ripple_threshold)

    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')

    rip_sec = pd.DataFrame(ripple_times, columns=['start_sec', 'end_sec'],
                        index=index)

    return rip_sec


def detect_ripple_candi_edited(time_x, lfp, samp_rate,
                               lo_hz=150,
                               hi_hz=250,
                               minimum_duration=0.015,
                               zscore_threshold=1.0,
                               smoothing_sigma=0.004,
                               close_ripple_threshold=0.0):

    filtered_lfps = np.stack(
        [filter_ripple_band(lfp, samp_rate, lo_hz=100, hi_hz=250) for lfp in lfp.T])

    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=0)

    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, samp_rate)

    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)

    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time_x, minimum_duration, zscore_threshold)

    return candidate_ripple_times



## Parameters
samp_rate = 1000


## Load
fpath = args.npy_fpath
lfp = np.load(fpath).squeeze().astype(np.float32)
lfp = lfp[:, np.newaxis] # The shape of LFP should be (len(lfp), 1) to fullfil the requirement of the ripple detector.

start_sec, end_sec, step_sec = 0, 512/1000, 1.0/samp_rate
# start_sec, end_sec, step_sec = 0, 1.*len(lfp)/samp_rate, 1.0/samp_rate
time_x = np.arange(start_sec, end_sec, step_sec)
lfp = lfp[int(start_sec*samp_rate):int(end_sec*samp_rate)]


## Detect Ripple Candidates
print('Detecting ripples from {} (Length: {:.1f}h)'.format(fpath, len(lfp)/samp_rate/3600))
%timeit rip_sec = detect_ripple_candi(time_x, lfp, samp_rate, lo_hz=150, hi_hz=250, zscore_threshold=1)
%timeit rip_sec = detect_ripple_candi_edited(time_x, lfp, samp_rate, lo_hz=150, hi_hz=250, zscore_threshold=1)


# ## Save
# savedir, fname, ext = mf.split_fpath(fpath)
# savepath = savedir + fname + '_ripples.pkl' # .format(sd_thresh)
# mf.pkl_save(rip_sec, savepath)


## EOF
