#!/usr/bin/env python

import argparse
import numpy as np
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
sys.path.append('05_Ripple/rippledetection/')
from core import (exclude_close_events, exclude_movement, filter_ripple_band,
                   gaussian_smooth, get_envelope,
                   get_multiunit_population_firing_rate,
                   merge_overlapping_ranges, threshold_by_zscore)
from scipy.stats import zscore
import pandas as pd
import myutils.myfunc as mf
plt.rcParams['font.size'] = 20 # 16
H, W = 10, 4
scale = 1.5
plt.rcParams["figure.figsize"] = (H*scale, W*scale) # (5, 2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
linewidth = 1


mytime = mf.time_tracker()

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath", default='../data/01/day1/split/1kHz/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## Funcs
def detect_ripple_candi(time_x, lfp, samp_rate,
                        lo_hz=150,
                        hi_hz=250,
                        minimum_duration=0.015,
                        zscore_threshold=1.0,
                        smoothing_sigma=0.004,
                        close_ripple_threshold=0.0):

    not_null = np.all(pd.notnull(lfp), axis=1)

    lfp, time_x = lfp[not_null], time_x[not_null]

    filtered_lfps = np.stack(
        [filter_ripple_band(lfp, samp_rate, lo_hz=100, hi_hz=250) for lfp in lfp.T])

    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=0)

    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, samp_rate)

    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)
    ripple_magni = combined_filtered_lfps.copy() # added

    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time_x, minimum_duration, zscore_threshold)

    ripple_times = exclude_close_events(
        candidate_ripple_times, close_ripple_threshold)

    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')

    rip_sec = pd.DataFrame(ripple_times, columns=['start_sec', 'end_sec'],
                        index=index)

    return filtered_lfps, ripple_magni, rip_sec


def parse_fpaths_emg(fpath_lfp):
  from glob import glob
  ttx_txt = 'tt3*' if '../data/02/' in fpath_lfp else 'tt8*'
  ldir, fname, ext = mf.split_fpath(fpath_lfp)
  emg_fpaths = glob(ldir + ttx_txt)
  return emg_fpaths


def load_ave_emg(fpaths_emg):
  emgs = []
  for f in fpaths_emg:
    emgs.append(np.load(f))
  emgs = np.array(emgs)
  emg_ave = emgs.mean(axis=0)
  return emg_ave


def calc_band_magnitude(data, samp_rate, lo_hz, hi_hz,
                        devide_by_std=False,
                        zscore_threshold=2.0,
                        smoothing_sigma=0.004,
                        close_ripple_threshold=0.0):

    if (lo_hz, hi_hz) != (None, None):
        filted = bandpass(data, lo_hz, hi_hz, samp_rate)
    else:
        filted = data

    power = filted ** 2
    smoothed_power = gaussian_smooth(power, smoothing_sigma, samp_rate)
    magnitude = np.sqrt(smoothed_power)

    if devide_by_std:
        magnitude /= magnitude.std() # Normalize

    return magnitude


## Parameters
SAMP_RATE = 1000

## Parse
fpath = args.npy_fpath
fpaths_emg = parse_fpaths_emg(fpath)


## Load
lfp = np.load(fpath).squeeze().astype(np.float32)
lfp = lfp[:, np.newaxis]
emg = load_ave_emg(fpaths_emg).astype(np.float32)


## Shortcut
start_sec, end_sec, step_sec = 0, 1.*3600, 1.0/SAMP_RATE # 1H
time_x = np.arange(start_sec, end_sec, step_sec)
lfp = lfp[int(start_sec*SAMP_RATE):int(end_sec*SAMP_RATE)]


## Detect Ripple Candidates
print('Detecting ripples from {} (Length: {:.1f}h)'.format(fpath, len(lfp)/SAMP_RATE/3600))
zscore_thres = 1
filted, ripple_magni, rip_sec = detect_ripple_candi(time_x,
                                                    lfp,
                                                    SAMP_RATE,
                                                    lo_hz=150,
                                                    hi_hz=250,
                                                    zscore_threshold=zscore_thres)
zscored_ripple_magni = zscore(ripple_magni)
mytime()


## Make Analog (0 or 1) Ripple Flags to Visualization
ripple_analog = np.zeros_like(time_x)
for i_rip in range(len(rip_sec)):
    ripple = rip_sec.iloc[i_rip]
    start_pts, end_pts = int(ripple['start_sec']*SAMP_RATE), int(ripple['end_sec']*SAMP_RATE)
    ripple_analog[start_pts:end_pts] = 1


## Plot
start_sec, end_sec = 1173, 1178
# Indexing for plotting
x = np.arange(start_sec, end_sec, step_sec)
lfp_plot = lfp[int(start_sec*SAMP_RATE):int(end_sec*SAMP_RATE)]
filted_plot = filted.squeeze()[int(start_sec*SAMP_RATE):int(end_sec*SAMP_RATE)]
zscored_ripple_magni_plot = zscored_ripple_magni.squeeze()[int(start_sec*SAMP_RATE):int(end_sec*SAMP_RATE)]
zscored_ripple_magni_std_thres_plot = zscore_thres*np.ones_like(x) # *zscored_ripple_magni.std()
zscored_ripple_magni_mean_plot = np.ones_like(x)*zscored_ripple_magni.mean()
ripple_analog_plot = ripple_analog[int(start_sec*SAMP_RATE):int(end_sec*SAMP_RATE)]
# Plot
fig, ax = plt.subplots(4,1, sharex=True)
ax[0].plot(x, lfp_plot, linewidth=linewidth)
ax[1].plot(x, filted_plot, linewidth=linewidth)
ax[2].plot(x, zscored_ripple_magni_plot, linewidth=linewidth)
ax[2].plot(x, zscored_ripple_magni_std_thres_plot,
           linewidth=linewidth, color='black', label='Zscore_Thres{}'.format(zscore_thres))
ax[2].plot(x, zscored_ripple_magni_mean_plot,
           linewidth=linewidth, color='black', label='Mean')
ax[3].plot(x, ripple_analog_plot, linewidth=linewidth)
# Limits
ax[0].set_ylim(-1000, 1000)
ax[1].set_ylim(-500, 500)
ax[2].set_ylim(-2, 10)
ax[2].legend()
## Ripple Coloring
rip_sec_plt = rip_sec[(start_sec < rip_sec['start_sec']) & (rip_sec['end_sec'] < end_sec )]
put_legend = False
for i in range(4):
    first = True
    for ripple in rip_sec_plt.itertuples():
        if first:
            label = 'Ripple Candi.' if put_legend else None
            ax[i].axvspan(ripple.start_sec, ripple.end_sec, alpha=0.1, color='red', zorder=1000, label=label)
            first = False
        else:
            ax[i].axvspan(ripple.start_sec, ripple.end_sec, alpha=0.1, color='red', zorder=1000)

# spath = '/home/ywatanabe/Desktop/slides/12/repr_traces_{}-{}s.tif'.format(start_sec, end_sec)
# fig.savefig(spath, dpi=300, format='tif')
'''
ax[0].legend()
ax[1].legend()
ax[2].legend()
'''



## Save
sdir = '../figs/paper/fig01/'
spath = sdir + 'how_to_define_ripple_candidates_{}-{}s.csv'.format(start_sec, end_sec)

# ax[0].plot(x, lfp_plot, linewidth=linewidth)
# ax[1].plot(x, filted_plot, linewidth=linewidth)
# ax[2].plot(x, zscored_ripple_magni_plot, linewidth=linewidth)
# ax[2].plot(x, zscored_ripple_magni_std_thres_plot,
#            linewidth=linewidth, color='black', label='Zscore_Thres{}'.format(zscore_thres))
# ax[2].plot(x, zscored_ripple_magni_mean_plot,
#            linewidth=linewidth, color='black', label='Mean')
# ax[3].plot(x, ripple_analog_plot, linewidth=linewidth)

df = pd.DataFrame({'lfp_plot':lfp_plot.squeeze(),
                   'filted_plot':filted_plot.squeeze(),
                   'zscored_ripple_magni_plot':zscored_ripple_magni_plot.squeeze(),
                   'zscored_ripple_magni_std_thres_plot':zscored_ripple_magni_std_thres_plot.squeeze(),
                   'zscored_ripple_magni_mean_plot':zscored_ripple_magni_mean_plot.squeeze(),
                   'ripple_analog_plot':ripple_analog_plot.squeeze(),
                  })
df.to_csv(spath)
print("Saved to {}".format(spath))


# start_sec, end_sec = 1161, 1163
# start_sec, end_sec = 1171, 1173
# start_sec, end_sec = 1005, 1007
# start_sec, end_sec = 1005, 1177
# start_sec, end_sec = 1164, 1165 # Two False Ripples
# start_sec, end_sec = 1171.5, 1172.5 # Two True Ripple

