#!/usr/bin/env python
import argparse
import sys; sys.path.append('.')
import numpy as np

import utils.general as ug
from ripples.EDA.funcs.calc_ripple_properties import calc_ripple_properties



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath",
                default='./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## PATHs
lpath_lfp = args.npy_fpath
lpath_ripples = lpath_lfp.replace('LFP_MEP_1kHz_npy', 'ripple_candi_1kHz_pkl')\
                         .replace('.npy', '.pkl')


# Gets parameters
SAMP_RATE = ug.get_samp_rate_int_from_fpath(lpath_lfp)


## Loads
rip_sec_df = ug.load_pkl(lpath_ripples)

## Renames columns
rip_sec_df['start_sec'] = rip_sec_df['start_time']
rip_sec_df['end_sec'] = rip_sec_df['end_time']
del rip_sec_df['start_time'], rip_sec_df['end_time'], rip_sec_df['duration']


## Saves
spath = lpath_ripples # overwrite
ug.save_pkl(rip_sec_df, spath)

## EOF
