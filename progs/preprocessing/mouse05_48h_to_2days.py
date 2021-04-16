#!/usr/bin/env python

'''
The recording for mouse#05 on "DAY4" was over 24 hours and it reached about 48 hours.
Thus, we decided to split the 48-hour file and make "DAY5" recording.
'''

import argparse
import numpy as np
import sys; sys.path.append('.')
from tqdm import tqdm
import os
import re

from progs.utils.general import (get_samp_rate_str_from_fpath,
                                 to_int_samp_rate,
                                 calc_h,
                                 split_fpath
                                 )
                           


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--numpy", default='./data/05/day4/split/LFP_MEP_1kHz_npy/1_tt5-3_fp16.npy',
                help="path to input .npy file")
args = ap.parse_args()


## Load
data1 = np.load(args.numpy)
data2 = np.load(args.numpy.replace('/1_', '/2_'))
data = np.concatenate((data1, data2), axis=0)


## Confirming lenghts
SAMP_RATE = to_int_samp_rate(get_samp_rate_str_from_fpath(args.numpy))
data1_h = calc_h(data1, SAMP_RATE)
data2_h = calc_h(data2, SAMP_RATE)
tot_h = data1_h + data2_h
print('Mouse #05 on original \"DAY4\" Total Hour: {:.1f}h'.format(tot_h))


## Split
data_day4 = data[:24*60*60*SAMP_RATE]
data_day5 = data[24*60*60*SAMP_RATE:]
print('DAY4 after splitting: {:.1f}h'.format(calc_h(data_day4, SAMP_RATE)))
print('DAY5 after splitting: {:.1f}h'.format(calc_h(data_day5, SAMP_RATE)))


## Save
# day4
spath_day4 = args.numpy.replace('/1_', '/')
np.save(spath_day4, data_day4)
print('Saved to: {}'.format(spath_day4))

# day5
spath_day5 = spath_day4.replace('day4', 'day5')
sdir_day5, _, _ = split_fpath(spath_day5)
os.makedirs(sdir_day5, exist_ok=True)
np.save(spath_day5, data_day5)
print('Saved to {}'.format(spath_day5))
print('DAY5: {:.1f}h'.format(calc_h(data_day5, SAMP_RATE)))


## EOF
