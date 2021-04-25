#!/usr/bin/env python
import argparse
import sys; sys.path.append('.')
import os

import utils.general as ug
from utils.EDA.calc_ripple_properties import calc_ripple_properties



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath",
                default='./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## PATHs
lpath_lfp = args.npy_fpath
lpath_rip = lpath_lfp.replace('LFP_MEP_1kHz_npy', 'ripple_candi_1kHz_pkl')\
                         .replace('.npy', '.pkl')


## Loads
rip_sec_df = ug.load_pkl(lpath_rip)


## Calculates ripple properties
rip_sec_with_props_df = calc_ripple_properties(rip_sec_df, lpath_lfp)


## Saves
spath = lpath_rip.replace('orig', 'with_props')
sdir, _, _ = ug.split_fpath(spath)
os.makedirs(sdir, exist_ok=True)
ug.save_pkl(rip_sec_with_props_df, spath)


## EOF
