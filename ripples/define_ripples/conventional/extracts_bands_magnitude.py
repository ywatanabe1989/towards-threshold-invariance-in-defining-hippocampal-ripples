#!/usr/bin/env python

import argparse
import sys

import numpy as np

sys.path.append(".")
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-l",
    "--lfp_fpath",
    default="./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy",
    help="The path of the input lfp file (.npy)",
)
args = ap.parse_args()


## Sets tee
sys.stdout, sys.stderr = utils.general.tee(sys)


## Parameters
SAMP_RATE = utils.pj.get_samp_rate_int_from_fpath(args.lfp_fpath)
RIPPLE_CANDI_LIM_HZ = utils.general.load("./conf/global.yaml")["RIPPLE_CANDI_LIM_HZ"]


## PATHs
LPATH_LFP = args.lfp_fpath
LPATHs_MEP = utils.pj.path_converters.LFP_to_MEPs(LPATH_LFP)


## Loads
lfp = np.load(LPATH_LFP).squeeze().astype(np.float32)[:, np.newaxis]
mep = (
    np.array([np.load(l).squeeze()[:, np.newaxis] for l in LPATHs_MEP])
    .mean(axis=0)
    .astype(np.float32)
)
assert len(lfp) == len(mep)


## Magnitudes
norm_mep_magni = utils.dsp.calc_band_magnitude(
    mep, SAMP_RATE, lo_hz=None, hi_hz=None, devide_by_std=True
).astype(np.float16)

norm_rip_magni = utils.dsp.calc_band_magnitude(
    lfp,
    SAMP_RATE,
    lo_hz=RIPPLE_CANDI_LIM_HZ[0],
    hi_hz=RIPPLE_CANDI_LIM_HZ[1],
    devide_by_std=True,
).astype(np.float16)


## Save
SPATH_NORM_MEP_MAGNI = utils.path_converters.LFP_to_MEP_magni(LPATH_LFP)
utils.general.save(norm_mep_magni, SPATH_NORM_MEP_MAGNI)

SPATH_NORM_RIP_MAGNI = utils.path_converters.LFP_to_ripple_magni(LPATH_LFP)
utils.general.save(norm_rip_magni, SPATH_NORM_RIP_MAGNI)

## EOF
