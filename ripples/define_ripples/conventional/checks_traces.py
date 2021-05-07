#!/usr/bin/env python3

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys; sys.path.append('.')
from ripples.define_ripples.conventional.defines_ripple_candidates import define_ripple_candidates
import utils.general as ug
import utils.path_converters as upcvt


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath",
                default='./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()



## PATHs
lpath_lfp = args.npy_fpath

## Loads
lfp = ug.load(lpath_lfp).squeeze()


## Parameters
samp_rate = ug.get_samp_rate_int_from_fpath(lpath_lfp)
step_sec = 1. / samp_rate


## Limit LFP regarding time
start_sec, end_sec = 0, 2000
time_x_sec = np.arange(start_sec, end_sec, step_sec)

start_pts, end_pts = int(start_sec*samp_rate), int(end_sec*samp_rate)
lfp = lfp[start_pts:end_pts]


## Detect Ripple Candidates
print('\nDetecting ripples from {} (Length: {:.1f}h\n)'.format(lpath_lfp, len(lfp)/samp_rate/3600))
RIPPLE_CANDI_LIM_HZ = ug.load('./conf/global.yaml')['RIPPLE_CANDI_LIM_HZ']
zscore_thres = 1
filted, ripple_magni, rip_sec = define_ripple_candidates(time_x_sec,
                                                         lfp[:, np.newaxis].astype(np.float32),
                                                         samp_rate,
                                                         lo_hz=RIPPLE_CANDI_LIM_HZ[0],
                                                         hi_hz=RIPPLE_CANDI_LIM_HZ[1],
                                                         zscore_threshold=zscore_thres,
                                                         ) # 150, 250


################################################################################
## Plots
################################################################################
ug.configure_mpl(plt)

# slicing
start_sec_plt = 1175.8
end_sec_plt = start_sec_plt + 2.2 # 1178.0
x_plt = np.arange(start_sec_plt+step_sec, end_sec_plt, step_sec)

ss = math.floor(start_sec_plt*samp_rate)
ee = math.floor(end_sec_plt*samp_rate)

lfp_plt = lfp[ss:ee]
filted_plt = filted.squeeze()[ss:ee]
ripple_magni_plt = ripple_magni.squeeze()[ss:ee]
ripple_magni_std_plt = np.ones_like(ripple_magni_plt) * ripple_magni.std()

# Plots lines
fig, ax = plt.subplots(3,1, sharex=True)
lw = 1
ax[0].plot(x_plt, lfp_plt, linewidth=lw, label='raw LFP')
label_filted = 'ripple band LFP (150-250 Hz)'.format(RIPPLE_CANDI_LIM_HZ[0], RIPPLE_CANDI_LIM_HZ[1])
ax[1].plot(x_plt, filted_plt, linewidth=lw, label=label_filted)
ax[2].plot(x_plt, ripple_magni_plt, linewidth=lw, label='ripple band magnitude')
ax[2].plot(x_plt, ripple_magni_std_plt, linewidth=lw, label='1 SD')

# Areas indicating ripple candidates
rip_sec_plt = rip_sec[(start_sec_plt < rip_sec['start_sec']) & (rip_sec['end_sec'] < end_sec_plt)]
for i in range(len(ax)):
    first = True
    for ripple in rip_sec_plt.itertuples():
        if first:
            ax[i].axvspan(ripple.start_sec, ripple.end_sec, alpha=0.1,
                          color='red', zorder=1000, label='Ripple candi.')
            first = False
        else:
            ax[i].axvspan(ripple.start_sec, ripple.end_sec, alpha=0.1,
                          color='red', zorder=1000)

ax[0].legend()
ax[0].set_ylabel('Amplitude [uV]')
ax[0].set_title('Defining Ripple Candidates')
ax[1].legend()
ax[1].set_ylabel('Amplitude [uV]')
ax[2].legend()
ax[2].set_ylabel('Amplitude [uV]')
ax[2].set_xlabel('Time [sec]')
# plt.show()
ug.save(plt, 'traces.tiff')


################################################################################
## Saves
################################################################################
df = pd.DataFrame({'lfp_plt':lfp_plt.squeeze(),
                   'filted_plt':filted_plt.squeeze(),
                   'ripple_magni_plt':ripple_magni_plt.squeeze(),
                   'ripple_magni_std_plt':ripple_magni_std_plt.squeeze(),
                  })
ug.save(df, 'traces.csv')


## EOF
