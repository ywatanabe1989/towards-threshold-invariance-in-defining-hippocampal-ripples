#!/usr/bin/env python

import argparse
import math
import numpy as np
import pandas as pd

import sys; sys.path.append('.')
import os
from progs.Fig_01_02_Ripple_candidates.utils.detect_ripple_candidates import detect_ripple_candidates
from progs.utils.general import (time_tracker,
                                 to_int_samp_rate,
                                 get_samp_rate_str_from_fpath,
                                 split_fpath,
                                 )


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath", default='./data/01/day1/split/LFP_MEP_1kHz_npy/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()



## PATHs
LPATH_LFP = args.npy_fpath
samp_rate = to_int_samp_rate(get_samp_rate_str_from_fpath(LPATH_LFP))


## Load LFP
# The shape of LFP should be (len(lfp), 1) to fullfil the requirement of the ripple detector.
lfp = np.load(LPATH_LFP).squeeze()[:, np.newaxis].astype(np.float32)


## Limit LFP regarding time
start_sec, end_sec = 0, 2000
step_sec = 1. / samp_rate
time_x_sec = np.arange(start_sec, end_sec, step_sec)

start_pts, end_pts = int(start_sec*samp_rate), int(end_sec*samp_rate)
lfp = lfp[start_pts:end_pts]


## Detect Ripple Candidates
print('Detecting ripples from {} (Length: {:.1f}h)'.format(LPATH_LFP, len(lfp)/samp_rate/3600))
filted, ripple_magni, rip_sec = detect_ripple_candidates(time_x_sec, lfp, samp_rate, lo_hz=150, hi_hz=250, zscore_threshold=1)



## Plot
plt.rcParams['font.size'] = 16
plt.rcParams["figure.figsize"] = (5, 2) # (36, 20)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

start_sec_plot = 1175.8
end_sec_plot = 1178.
x_plot = np.arange(start_sec_plot+step_sec, end_sec_plot, step_sec)


lfp_plot = lfp[math.floor(start_sec_plot*samp_rate):math.floor(end_sec_plot*samp_rate)]
filted_plot = filted.squeeze()[math.floor(start_sec_plot*samp_rate):math.floor(end_sec_plot*samp_rate)]
ripple_magni_plot = ripple_magni.squeeze()[math.floor(start_sec_plot*samp_rate):math.floor(end_sec_plot*samp_rate)]
ripple_magni_std_plot = np.ones_like(ripple_magni_plot)*ripple_magni.std()


fig, ax = plt.subplots(3,1, sharex=True)
linewidth = 1
ax[0].plot(x_plot, lfp_plot, linewidth=linewidth)
ax[1].plot(x_plot, filted_plot, linewidth=linewidth)
ax[2].plot(x_plot, ripple_magni_plot, linewidth=linewidth)
ax[2].plot(x_plot, ripple_magni_std_plot, linewidth=linewidth)


# area indicating ripple candidates
rip_sec_plt = rip_sec[(start_sec_plot < rip_sec['start_sec']) & (rip_sec['end_sec'] < end_sec_plot)]
for i in range(3):
    first = True
    for ripple in rip_sec_plt.itertuples():
        if first:
            ax[i].axvspan(ripple.start_sec, ripple.end_sec, alpha=0.1, color='red', zorder=1000, label='Ripple candi.')
            first = False
        else:
            ax[i].axvspan(ripple.start_sec, ripple.end_sec, alpha=0.1, color='red', zorder=1000)

# plt.show()


## Save
sdir = './results/fig01/'
spath_csv = sdir + 'plot.csv'

sdirsplit_fpath(spath_csv)
df = pd.DataFrame({'lfp_plot':lfp_plot.squeeze(),
                   'filted_plot':filted_plot.squeeze(),
                   'ripple_magni_plot':ripple_magni_plot.squeeze(),
                   'ripple_magni_std_plot':ripple_magni_std_plot.squeeze(),
                  })
df.to_csv(spath_csv)
print('Saved to: {}'.format(spath_csv))

spath_tiff = spath_csv.replace('.csv', '.tiff')
plt.savefig(spath_tiff, dpi=300, format='tiff')


# df.to_csv(spath)

#                    'ripple_magni_std5_plot':ripple_magni_std5_plot.squeeze(),
