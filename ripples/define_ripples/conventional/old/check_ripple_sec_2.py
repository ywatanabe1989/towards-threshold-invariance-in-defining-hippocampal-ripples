#!/usr/bin/env python
import argparse
import sys; sys.path.append('.')
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

import utils.general as ug
from ripples.EDA.funcs.calc_ripple_properties import calc_ripple_properties



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath",
                default='./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## PATHs
lpath_lfp = args.npy_fpath
# './data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy'

# ./data/okada/from_nas1/01/day1/split/1kHz/tt2-1_fp16_ripple_candi_150-250Hz.pkl
lpath_ripples = lpath_lfp.replace('okada/', 'okada/from_nas1/')\
                         .replace('LFP_MEP_1kHz_npy/orig/', '1kHz/')\
                         .replace('.npy', '_ripple_candi_150-250Hz.pkl')


SAMP_RATE = ug.to_int_samp_rate(ug.get_samp_rate_str_from_fpath(lpath_lfp))

## Loads
lfp = np.load(lpath_lfp).squeeze()
ripples_sec_df = ug.load_pkl(lpath_ripples)[['start_sec', 'end_sec']]



## Checks lengths of LFP and time stamps of Ripple candidates.
lfp_len_sec = len(lfp) / SAMP_RATE
last_ripples_end_sec = ripples_sec_df.iloc[-1]['end_sec']
print('\n--------------------------------------------------------------------------------')
print(lpath_lfp)
print(lpath_ripples)
print(lfp_len_sec, last_ripples_end_sec)
print('--------------------------------------------------------------------------------\n')
# print(lpath_ripples)
# print(ripples_sec_df.head(3))


## EOF
