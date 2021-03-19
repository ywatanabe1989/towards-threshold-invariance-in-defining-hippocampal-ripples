import numpy as np
from ripple_detection import Kay_ripple_detector
import argparse
import time
import numpy as np
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
sys.path.append('05_Ripple/rippledetection/')
import myutils.myfunc as mf
from core import (exclude_close_events, exclude_movement, filter_ripple_band,
                   gaussian_smooth, get_envelope,
                   get_multiunit_population_firing_rate,
                   merge_overlapping_ranges, threshold_by_zscore)
from tqdm import tqdm
import seaborn as sns
import time
import pandas as pd
from bisect import bisect_left, bisect_right



## Funcs
def detect_ripple_candi(time_x, lfp, samp_rate,
                        lo_hz=150,
                        hi_hz=250,
                        minimum_duration=0.015,
                        zscore_threshold=1.0,
                        smoothing_sigma=0.004,
                        close_ripple_threshold=0.0):

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
