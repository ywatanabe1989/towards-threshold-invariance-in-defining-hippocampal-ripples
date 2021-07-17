#!/usr/bin/env python

import argparse
import os
import sys

sys.path.append(".")

import numpy as np
import pandas as pd
import utils

ts = utils.general.TimeStamper()

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-n",
    "--npy_fpath",
    default="./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy",
    help="The path of the input lfp file (.npy)",
)
args = ap.parse_args()


## Sets tee
sys.stdout, sys.stderr = utils.general.tee(sys)


## Parameters
samp_rate = utils.pj.get_samp_rate_int_from_fpath(args.npy_fpath)


## Loads
fpath = args.npy_fpath
lfp = np.load(fpath).squeeze().astype(np.float32)
lfp = lfp[:, np.newaxis]

start_sec, end_sec, step_sec = 0, 1.0 * len(lfp) / samp_rate, 1.0 / samp_rate
time_x = np.arange(start_sec, end_sec, step_sec)


## Detects Ripple Candidates
print(
    "\nDetecting ripples from {} (Length: {:.1f}h\n)".format(
        fpath, len(lfp) / samp_rate / 3600
    )
)

lo_hz_ripple, hi_hz_ripple = utils.general.load("./conf/global.yaml")[
    "RIPPLE_CANDI_LIM_HZ"
]
_, _, rip_sec = utils.pj.define_ripple_candidates(
    time_x, lfp, samp_rate, lo_hz=lo_hz_ripple, hi_hz=hi_hz_ripple, zscore_threshold=1
)
ts("")


## Save the ripple candidates
ldir, fname, ext = utils.general.split_fpath(fpath)
sdir = (
    ldir.replace("LFP_MEP", "ripples")
    .replace("/orig/", "/candi_orig/")
    .replace("_npy", "_pkl")
)
spath = sdir + fname + ".pkl"
utils.general.save(rip_sec, spath)
# Saved to: './data/okada/01/day1/split/ripples_1kHz_pkl/candi_orig/tt2-1_fp16.pkl'

## EOF
