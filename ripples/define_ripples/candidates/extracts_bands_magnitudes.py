#!/usr/bin/env python

import argparse
import numpy as np
from glob import glob


import sys; sys.path.append('.')
import utils.general as ug
import utils.dsp as ud


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-l", "--lfp_fpath",
                default='./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()



## Parameters
SAMP_RATE = ug.get_samp_rate_int_from_fpath(args.lfp_fpath)
LOW_HZ_RIPPLE, HIGH_HZ_RIPPLE = 150, 250


## PATHs
LPATH_LFP = args.lfp_fpath
_LDIR, _, _ = ug.split_fpath(LPATH_LFP)
_FPATHS_TRAPE_MEP = ug.read_txt('./data/okada/FPATH_LISTS/TRAPE_MEP_TT_NPYs.txt')
LPATHs_MEP = ug.search_str_list(_FPATHS_TRAPE_MEP, _LDIR)[1]


## Load
lfp = np.load(LPATH_LFP).squeeze().astype(np.float32)[:, np.newaxis]
mep = np.array([np.load(l).squeeze()[:, np.newaxis] for l in LPATHs_MEP])\
        .mean(axis=0).astype(np.float32)
assert len(lfp) == len(mep)


## Magnitudes
mep_magni_sd = ud.calc_band_magnitude(mep, SAMP_RATE,
                                   lo_hz=None, hi_hz=None, devide_by_std=True).astype(np.float16)
ripple_band_magni_sd = ud.calc_band_magnitude(lfp, SAMP_RATE,
                lo_hz=LOW_HZ_RIPPLE, hi_hz=HIGH_HZ_RIPPLE, devide_by_std=True).astype(np.float16)


## Save
SPATH_MEP_MAGNI_SD = LPATH_LFP.replace('orig', 'magni')\
                              .replace('_fp16.npy', '_mep_magni_sd_fp16.npy')
SPATH_RIPPLE_BAND_MAGNI = LPATH_LFP.replace('orig', 'magni')\
                                   .replace('_fp16.npy', '_ripple_band_magni_sd_fp16.npy')

ug.save_npy(mep_magni_sd, SPATH_MEP_MAGNI_SD)
ug.save_npy(ripple_band_magni_sd, SPATH_RIPPLE_BAND_MAGNI)

## EOF
