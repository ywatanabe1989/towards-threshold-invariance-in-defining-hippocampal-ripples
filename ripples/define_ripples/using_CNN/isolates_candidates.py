#!/usr/bin/env python
import argparse
from functools import partial
import sys
sys.path.append('.')
import math
import numpy as np
import pandas as pd


import utils.general as ug
import utils.semi_ripple as us
import utils.path_converters as upcvt



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-l", "--lpath_lfp",
                default='./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## Functions
def isolates_rip_row(rip_row, lfp=None, max_duration_pts=400):
    # padding
    if rip_row['duration_pts'] < max_duration_pts:
        '''
        rip_row = rips.iloc[0]
        '''

        pad_all = max_duration_pts - rip_row['duration_pts']
        
        pad1, pad2 = math.floor(pad_all/2), math.floor(pad_all/2)
        if pad_all % 2 == 1:
            pad1 += 1
        
        start_to_rip_peak = rip_row['ripple_peak_posi_pts'] - rip_row['start_pts']
        rip_peak_to_end = rip_row['end_pts'] - rip_row['ripple_peak_posi_pts']

        if int(max_duration_pts/2) < start_to_rip_peak:
            pad1 = 0
            pad2 = pad_all

        elif int(max_duration_pts/2) < rip_peak_to_end:
            pad1 = pad_all
            pad2 = 0

        else:
            delta = start_to_rip_peak - rip_peak_to_end

            if delta < 0:
                pad1 += abs(int(delta/2))
                pad2 -= abs(int(delta/2))

            if delta > 0:
                pad1 -= abs(int(delta/2))
                pad2 += abs(int(delta/2))

            if pad1 <= 0:
                pad1 = 0
                pad2 -= abs(pad1)

            if pad2 <= 0:
                pad1 -= abs(pad2)
                pad2 = 0

        # # for debugging
        # padded_len = np.array([pad1, start_to_rip_peak, rip_peak_to_end, pad2]).sum()
        # print(padded_len)
        # if not padded_len == max_duration_pts:
        #     import pdb; pdb.set_trace()
        #     print(pad1, start_to_rip_peak, rip_peak_to_end, pad2)

        padded = np.pad(rip_row['LFP'], [pad1, pad2], 'constant', constant_values=(0))
        assert len(padded) == max_duration_pts
        isolated = padded # alias

    # trimming
    else:
        start = rip_row['ripple_peak_posi_pts'] - 200
        end = rip_row['ripple_peak_posi_pts'] + 200        
        trimmed = lfp[start:end]

        len_trimmed = len(trimmed)
        if len_trimmed < max_duration_pts: # pad
            pad_all = max_duration_pts - len_trimmed
            pad1, pad2 = math.floor(pad_all/2), math.floor(pad_all/2)
            if pad_all % 2 == 1:
                pad1 += 1
            trimmed = np.pad(trimmed, [pad1, pad2], 'constant', constant_values=(0))

        isolated = trimmed # alias

    assert len(isolated) == max_duration_pts
    
    return isolated



## FPATHs
lpath_lfp = args.lpath_lfp
lpath_rip = upcvt.LFP_to_ripples(lpath_lfp, rip_sec_ver='candi_orig')
lpath_rip_magni = upcvt.LFP_to_ripple_magni(lpath_lfp)


## Loads
lfp = ug.load(lpath_lfp).squeeze()
rips = ug.load(lpath_rip)
rip_magni = ug.load(lpath_rip_magni).squeeze()


## Gets Parameters
SAMP_RATE = ug.get_samp_rate_int_from_fpath(lpath_lfp) # 1000


## Isolates each ripple candidate
# Preparation
rips['start_pts'] = (rips['start_sec'] * SAMP_RATE).astype(int)
rips['end_pts'] = (rips['end_sec'] * SAMP_RATE).astype(int)
rips['duration_pts'] = rips['end_pts'] - rips['start_pts']
rips['LFP'] = [lfp[int(rip['start_pts']):int(rip['end_pts'])]
               for i_rip, rip in rips.iterrows()]
rips['Ripple_Band_Magnitude'] = [rip_magni[int(rip['start_pts']):int(rip['end_pts'])]
                                 for i_rip, rip in rips.iterrows()]
rips['ripple_peak_posi_pts'] = [int(rip['start_pts']) + rip['Ripple_Band_Magnitude'].argmax()
                                for i_rip, rip in rips.iterrows()]

# Isolates each ripple candidate
# print(isolates_rip_row(rips.iloc[0], lfp=lfp))
isolates_rip_row_par = partial(isolates_rip_row, lfp=lfp)
# print(isolates_row_par(rips.iloc[0]))
rips['isolated'] = rips.apply(isolates_rip_row_par, axis=1)


## Excludes unnecessary columns
keys_to_del = ['start_pts',
               'end_pts',
               'duration_pts',
               'LFP',
               'Ripple_Band_Magnitude',
               'ripple_peak_posi_pts',
               ]
for k in keys_to_del:
    del rips[k]


## Saves
spath_rip = lpath_rip.replace('candi_orig', 'isolated')
ug.save(rips, spath_rip)


## EOF
