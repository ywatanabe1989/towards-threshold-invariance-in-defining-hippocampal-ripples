#!/usr/bin/env python

import argparse
import numpy as np
import time
from tqdm import tqdm
import seaborn as sns
import pandas as pd

import sys; sys.path.append('.')

from progs.Fig_01_02_Ripple_candidates.utils.rippledetection.core import (exclude_close_events,
                  filter_ripple_band,
                  gaussian_smooth,
                  threshold_by_zscore)

# sys.path.append('05_Ripple/')
# sys.path.append('05_Ripple/rippledetection/')
# import myutils.myfunc as mf
# from core import (exclude_close_events,
#                   filter_ripple_band,
#                   gaussian_smooth,
#                   threshold_by_zscore)
# from ripple_detection import Kay_ripple_detector


## Functions
def detect_ripple_candidates(time_x, lfp, samp_rate,
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
