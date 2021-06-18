#!/usr/bin/env python

import numpy as np
import pandas as pd
from modules.rippledetection.core import (exclude_close_events, filter_band,
                                          gaussian_smooth, threshold_by_zscore)

# def calc_band_magnitude(
#     data,
#     samp_rate,
#     lo_hz,
#     hi_hz,
#     devide_by_std=False,
#     smoothing_sigma=0.004,
# ):

#     if (lo_hz, hi_hz) != (None, None):
#         filted = bandpass(data, lo_hz, hi_hz, samp_rate)
#     else:
#         filted = data

#     power = filted ** 2
#     smoothed_power = gaussian_smooth(power, smoothing_sigma, samp_rate)
#     magnitude = np.sqrt(smoothed_power)

#     if devide_by_std:
#         magnitude /= magnitude.std()  # Normalize

#     return magnitude


def calc_band_magnitude(
    data,
    samp_rate,
    lo_hz,
    hi_hz,
    devide_by_std=False,
    smoothing_sigma=0.004,
):
    if (lo_hz, hi_hz) != (None, None):  # Ripple band
        _, filted_magni, _ = define_ripple_candidates(
            None,
            data,
            samp_rate,
            lo_hz=lo_hz,
            hi_hz=hi_hz,
            smoothing_sigma=smoothing_sigma,
            only_calc_filted_magni=True,
        )
        # filted = bandpass(data, lo_hz, hi_hz, samp_rate)
    else:  # MEP
        filted = data
        power = filted ** 2
        smoothed_power = gaussian_smooth(power, smoothing_sigma, samp_rate)
        filted_magni = np.sqrt(smoothed_power).squeeze()

    if devide_by_std:
        filted_magni /= filted_magni.std()  # Normalize

    return filted_magni


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


def bandpass(data, lo_hz, hi_hz, fs, order=5):
    def mk_butter_bandpass(order=5):
        from scipy.signal import butter, sosfilt, sosfreqz

        nyq = 0.5 * fs
        low, high = lo_hz / nyq, hi_hz / nyq
        sos = butter(order, [low, high], analog=False, btype="band", output="sos")
        return sos

    def butter_bandpass_filter(data):
        from scipy.signal import butter, sosfilt, sosfreqz

        sos = mk_butter_bandpass()
        y = sosfilt(sos, data)
        return y

    sos = mk_butter_bandpass(order=order)
    y = butter_bandpass_filter(data)

    return y


def wavelet(wave, samp_rate, f_min=100, f_max=None, plot=False):
    dt = 1.0 / samp_rate
    npts = len(wave)
    t = np.linspace(0, dt * npts, npts)
    if f_min == None:
        f_min = 0.1
    if f_max == None:
        f_max = int(samp_rate / 2)
    scalogram = cwt(wave, dt, 8, f_min, f_max)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y = np.meshgrid(
            t, np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0])
        )

        ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_yscale("log")
        ax.set_ylim(f_min, f_max)
        plt.show()

    Hz = pd.Series(np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
    df = pd.DataFrame(np.abs(scalogram))
    df["Hz"] = Hz
    df.set_index("Hz", inplace=True)

    return df
