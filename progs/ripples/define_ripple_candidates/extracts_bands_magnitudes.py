#!/usr/bin/env python

import argparse
import sys; sys.path.append('.')
from progs.Fig_01_02_Ripple_candidates.utils.rippledetection.core import gaussian_smooth, get_envelope
from progs.utils.general import (get_samp_rate_str_from_fpath,
                                 to_int_samp_rate,
                                 read_txt,
                                 split_fpath,
                                 search_str_list,
                                 save_npy,
                                 )
from progs.utils.dsp import bandpass
import numpy as np
from glob import glob


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-l", "--lfp_fpath", default='./data/01/day1/split/LFP_MEP_1kHz_npy/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)") # './data/01/day1/split/LFP_MEP_1kHz_npy/tt2-1_fp16.npy'
args = ap.parse_args()


## Funcs
def calc_band_magnitude(data, samp_rate, lo_hz, hi_hz,
                        devide_by_std=False,
                        minimum_duration=0.15,
                        zscore_threshold=2.0,
                        smoothing_sigma=0.004,
                        close_ripple_threshold=0.0):

    if (lo_hz, hi_hz) != (None, None):
        filted = bandpass(data, lo_hz, hi_hz, samp_rate)
    else:
        filted = data

    power = filted ** 2
    smoothed_power = gaussian_smooth(power, smoothing_sigma, samp_rate)
    magnitude = np.sqrt(smoothed_power)

    if devide_by_std:
        magnitude /= magnitude.std() # Normalize

    return magnitude



## Parameters
SAMP_RATE = to_int_samp_rate(get_samp_rate_str_from_fpath(args.lfp_fpath))
LOW_HZ_RIPPLE, HIGH_HZ_RIPPLE = 150, 250


## PATHs
LPATH_LFP = args.lfp_fpath
_LDIR, _, _ = split_fpath(LPATH_LFP)
_FPATHS_TRAPE_MEP = read_txt('./data/TRAPE_MEP_TT_IDs.txt')
LPATHs_MEP = search_str_list(_FPATHS_TRAPE_MEP, _LDIR)[1]

# fixme; instead of _st, is it better to name as normalized ~ ?
SPATH_MEP_MAGNI_SD = LPATH_LFP.replace('_fp16.npy', '_mep_magni_sd_fp16.npy')
# lpath_sw_magni_sd = LPATH_LFP.replace('_fp16.npy', '_sw_magni_sd_fp16.npy')
# lpath_gamma_magni_sd = LPATH_LFP.replace('_fp16.npy', '_gamma_magni_sd_fp16.npy')
SPATH_RIPPLE_BAND_MAGNI = LPATH_LFP.replace('_fp16.npy', '_ripple_band_magni_sd_fp16.npy')


## Load
lfp = np.load(LPATH_LFP).squeeze().astype(np.float32)[:, np.newaxis]
mep = np.array([np.load(l).squeeze()[:, np.newaxis] for l in LPATHs_MEP]).mean(axis=0).astype(np.float32)
assert len(lfp) == len(mep)


## Magnitudes
mep_magni_sd = calc_band_magnitude(mep, SAMP_RATE, lo_hz=None, hi_hz=None, devide_by_std=True).astype(np.float16)
# sw_band_magni_sd = calc_band_magnitude(lfp, SAMP_RATE, lo_hz=LOW_HZ_SW, hi_hz=HIGH_HZ_SW, devide_by_std=True).astype(np.float16)
# gamma_band_magni_sd = calc_band_magnitude(lfp, SAMP_RATE, lo_hz=LOW_HZ_GAMMA, hi_hz=HIGH_HZ_GAMMA, devide_by_std=True).astype(np.float16)
ripple_band_magni_sd = calc_band_magnitude(lfp, SAMP_RATE, lo_hz=LOW_HZ_RIPPLE, hi_hz=HIGH_HZ_RIPPLE, devide_by_std=True).astype(np.float16)


## Save
save_npy(mep_magni_sd, SPATH_MEP_MAGNI_SD)
# mf.save_npy(sw_band_magni_sd, fpath_sw_band_magni_sd)
# mf.save_npy(gamma_band_magni_sd, fpath_gamma_band_magni_sd)
save_npy(ripple_band_magni_sd, SPATH_RIPPLE_BAND_MAGNI)

# Saved to: ./data/01/day1/split/LFP_MEP_1kHz_npy/tt2-1_mep_magni_sd_fp16.npy
# Saved to: ./data/01/day1/split/LFP_MEP_1kHz_npy/tt2-1_ripple_band_magni_sd_fp16.npy

## EOF