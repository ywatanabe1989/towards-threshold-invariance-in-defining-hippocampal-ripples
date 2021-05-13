#!/usr/bin/env python
import argparse
import sys; sys.path.append('.')
import os

import utils
# import utils.general as ug
from utils.EDA_funcs.calc_ripple_properties import calc_ripple_properties



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath",
                default='./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## PATHs
lpath_lfp = args.npy_fpath


## Calculates ripple properties
rip_sec_with_props_df = calc_ripple_properties(lpath_lfp)


## Saves
spath = utils.path_converters.LFP_to_ripples(lpath_lfp, rip_sec_ver='candi_with_props')
sdir, _, _ = utils.general.split_fpath(spath)
os.makedirs(sdir, exist_ok=True)
utils.general.save(rip_sec_with_props_df, spath)


## EOF
