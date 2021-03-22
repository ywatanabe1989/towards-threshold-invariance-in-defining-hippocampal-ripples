#!/usr/bin/env python

import argparse
import numpy as np
import os

import sys; sys.path.append('.')
from progs.Fig_01_02_Ripple_candidates.utils.detect_ripple_candidates import detect_ripple_candidates
from progs.utils.general import (time_tracker,
                                 split_fpath,
                                 save_pkl,
                                 to_int_samp_rate,
                                 get_samp_rate_str_from_fpath,
                                 )
                           

mytime = time_tracker()

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath", default='./data/01/day1/split/1kHz_npy/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## Parameters
samp_rate = to_int_samp_rate(get_samp_rate_str_from_fpath(args.npy_fpath))
# sd_thresh = args.sd_thresh


## Load
fpath = args.npy_fpath
lfp = np.load(fpath).squeeze().astype(np.float32)
lfp = lfp[:, np.newaxis] # The shape of LFP should be (len(lfp), 1) to fullfil the requirement of the ripple detector.

start_sec, end_sec, step_sec = 0, 1.*len(lfp)/samp_rate, 1.0/samp_rate
time_x = np.arange(start_sec, end_sec, step_sec)
lfp = lfp[int(start_sec*samp_rate):int(end_sec*samp_rate)]


## Detect Ripple Candidates
print('Detecting ripples from {} (Length: {:.1f}h)'.format(fpath, len(lfp)/samp_rate/3600))
rip_sec = detect_ripple_candidates(time_x, lfp, samp_rate, lo_hz=100, hi_hz=250, zscore_threshold=1)
mytime()


## Save
ldir, fname, ext = split_fpath(fpath)
sdir = ldir.replace('LFP_MEP', 'ripple_candi').replace('npy', 'pkl')
# sdir = ldir.replace('1kHz_npy', '1kHz_ripple_candi_pkl')
spath = sdir + fname + '.pkl' # .format(sd_thresh)
os.makedirs(sdir, exist_ok=True)
save_pkl(rip_sec, spath)


## EOF
