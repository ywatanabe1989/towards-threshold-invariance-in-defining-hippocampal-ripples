#!/usr/bin/env python3
# Time-stamp: "2021-09-16 10:28:17 (ywatanabe)"

#!/usr/bin/env python

import numpy as np
import pandas as pd
from ripple_detector_CNN.external.modified_ripple_detection.core import (
    exclude_close_events,
    filter_band,
    gaussian_smooth,
    threshold_by_zscore,
)


def define_ripple_candidates(
    time_x,
    lfp,
    samp_rate,
    lo_hz=150,
    hi_hz=250,
    minimum_duration=0.015,
    zscore_threshold=1.0,
    smoothing_sigma=0.004,
    close_ripple_threshold=0.0,
    only_calc_filted_magni=False,
):

    ## Checks signal shape
    if lfp.ndim == 1:
        lfp = lfp[:, np.newaxis]
    assert lfp.ndim == 2

    ## Checks signal dtype
    if lfp.dtype == np.float16:
        lfp = lfp.astype(np.float32)

    ## Checks NaN
    not_null = np.all(pd.notnull(lfp), axis=1)
    lfp = lfp[not_null]
    if not only_calc_filted_magni:
        time_x = time_x[not_null]

    ## Band-pass filtering
    print("\nRipple Band {}-{} Hz\n".format(lo_hz, hi_hz))
    filtered_lfps = np.stack(
        [filter_band(lfp, samp_rate, lo_hz=lo_hz, hi_hz=hi_hz) for lfp in lfp.T]
    )

    ## Sum over electrodes
    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=0)

    ## Gaussian filtering along with the time axis
    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, samp_rate
    )
    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)
    filted_magni = combined_filtered_lfps  # alias

    if only_calc_filted_magni:
        filtered_lfps = None
        rip_sec = None

    else:
        candidate_ripple_times = threshold_by_zscore(
            combined_filtered_lfps, time_x, minimum_duration, zscore_threshold
        )

        ripple_times = exclude_close_events(
            candidate_ripple_times, close_ripple_threshold
        )

        index = pd.Index(np.arange(len(ripple_times)) + 1, name="ripple_number")

        rip_sec = pd.DataFrame(
            ripple_times, columns=["start_sec", "end_sec"], index=index
        )

        filtered_lfps = filtered_lfps.squeeze()

    return filtered_lfps, filted_magni, rip_sec
