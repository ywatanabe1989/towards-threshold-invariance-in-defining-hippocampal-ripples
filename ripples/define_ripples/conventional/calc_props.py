#!/usr/bin/env python

import argparse
import sys

sys.path.append(".")
import os

import utils

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


## PATHs
lpath_lfp = args.npy_fpath


## Calculates ripple properties
rip_sec_with_props_df = utils.pj.calc_ripple_properties(lpath_lfp)


## Saves
spath = utils.pj.path_converters.LFP_to_ripples(
    lpath_lfp, rip_sec_ver="candi_with_props"
)
utils.general.save(rip_sec_with_props_df, spath)


## EOF
