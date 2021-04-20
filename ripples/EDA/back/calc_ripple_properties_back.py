#!/usr/bin/env python
import argparse
import sys; sys.path.append('.')
# sys.path.append('05_Ripple/')
# sys.path.append('05_Ripple/rippledetection/')
import numpy as np
# import myutils.myfunc as mf
from glob import glob
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

from utils.general import (load_pkl,
                           take_closest,
                           save_pkl,
                           get_samp_rate_str_from_fpath,
                           to_int_samp_rate,
                           )
from utils.dsp import (wavelet,
                       )



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath",
                default='./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## Parameters
SAMP_RATE = to_int_samp_rate(get_samp_rate_str_from_fpath(args.npy_fpath)) #1000


## Parse File Paths
lpath_lfp = args.npy_fpath
lpath_mep_magni = lpath_lfp.replace('orig', 'magni')\
                           .replace('_fp16.npy', '_mep_magni_sd_fp16.npy')
lpath_ripple_magni = lpath_lfp.replace('orig', 'magni')\
                              .replace('_fp16.npy', '_ripple_band_magni_sd_fp16.npy')

# lpath_mep_magni_sd = lpath_lfp.replace('_fp16.npy', '_mep_magni_sd_fp16.npy')
# lpath_ripple_magni_sd = lpath_lfp.replace('_fp16.npy', '_ripple_magni_sd_fp16.npy')

# fpath_ripples = lpath_lfp.replace('.npy', '_ripples.pkl')
lpath_ripples = lpath_lfp.replace('orig/', '')\
                         .replace('LFP_MEP_1kHz_npy', 'ripple_candi_1kHz_pkl')\
                         .replace('.npy', '.pkl')


## Load
lfp = np.load(lpath_lfp).squeeze().astype(np.float32).squeeze()
ripples_sec_df = load_pkl(lpath_ripples)[['start_time', 'end_time', 'duration']]
mep_magni = np.load(lpath_mep_magni).astype(np.float32).squeeze()
ripple_magni = np.load(lpath_ripple_magni).astype(np.float32).squeeze()



## Packs sliced LFP, MEP magnitude, and ripple band magnitude into one dataframe
df = ripples_sec_df.copy()
df['start_pts'] = (df['start_time']*SAMP_RATE).astype(int)
df['end_pts'] = (df['end_time']*SAMP_RATE).astype(int)
df['duration_ms'] = (df['duration']*1000).astype(int)
del df['start_time'], df['end_time'], df['duration']

df['LFP'] = [lfp[start_i:end_i]
             for start_i, end_i in
             zip (df['start_pts'], df['end_pts'])
             ]

df['MEP MAGNI.'] = [mep_magni[start_i:end_i]
                    for start_i, end_i in
                    zip (df['start_pts'], df['end_pts'])
                    ]

df['RIPPLE MAGNI.'] = [ripple_magni[start_i:end_i]
                       for start_i, end_i in
                       zip (df['start_pts'], df['end_pts'])
                       ]

## Calculates Properties during ripple candidates
df['ln(Duration)'] = np.log(df['duration_ms'])
df['mean ln(MEP MAGNI.)'] = np.log(df['MEP MAGNI.'].apply(np.mean))
df['ln(RIPPLE Peak MAGNI.)'] = np.log(df['RIPPLE MAGNI.'].apply(np.max))




# mep_magni_sd = np.load(lpath_mep_magni_sd).astype(np.float32).squeeze()
# sw_magni_sd = np.load(fpath_sw_magni_sd).astype(np.float32).squeeze()
# gamma_magni_sd = np.load(fpath_gamma_magni_sd).astype(np.float32).squeeze()
# ripple_magni_sd = np.load(lpath_ripple_magni_sd).astype(np.float32).squeeze()

# SD_MEP_MAGNI = (mep_magni / mep_magni_sd).mean()
# SD_SW_MAGNI = (sw_magni / sw_magni_sd).mean()
# SD_GAMMA_MAGNI = (gamma_magni / gamma_magni_sd).mean()
# SD_RIPPLE_MAGNI = (ripple_magni / ripple_magni_sd).mean()


## Main Loop
ripples['duration_ms'] = (ripples['end_sec'] - ripples['start_sec']) * 1000
sw_height = []
ripple_peaks_pos_sec = []
ripple_peaks_magnis = []
ripple_peaks_freq_hz = []
gamma_ave_magnis = []
emg_ave_magnis = []
for i_rip in range(len(ripples)):
    _start_pts, _end_pts = ripples.iloc[i_rip][['start_sec', 'end_sec']].T * SAMP_RATE

    _lfp_sliced = lfp[int(_start_pts):int(_end_pts)]
    _ripple_magni_sliced = ripple_magni[int(_start_pts):int(_end_pts)]
    _sw_magni_sliced = sw_magni[int(_start_pts):int(_end_pts)]
    _mep_magni_sliced = mep_magni[int(_start_pts):int(_end_pts)]
    _gamma_magni_sliced = gamma_magni[int(_start_pts):int(_end_pts)]


    ## Ripple Band
    # Magnitude
    ripple_peak_magni = _ripple_magni_sliced.max()
    # Peak Position
    _ripple_peak_pos_in_slice_pts = _ripple_magni_sliced.argmax()
    ripple_peak_pos_sec =  (_ripple_peak_pos_in_slice_pts + _start_pts) / SAMP_RATE

    # Peak Frequency
    # Wavelet
    _wavelet_out = wavelet(_lfp_sliced, SAMP_RATE, f_min=100, f_max=250, plot=False)
    _col, _row = np.where(_wavelet_out == _wavelet_out.max().max())
    _wavelet_out.iloc[_col, _row]
    ripple_peak_freq_hz = float(np.array(_wavelet_out.index[_col]))

    ## SW Band
    # Height
    try:
        _sw_peaks_pos_pts, _sw_peaks_props = find_peaks(_sw_magni_sliced, height=1e-10)
        _, _closest_sw_peak_idx = take_closest(_sw_peaks_pos_pts, _ripple_peak_pos_in_slice_pts)
        _closest_sw_peak_idx = np.clip(_closest_sw_peak_idx, 0, len(_sw_peaks_props['peak_heights'])-1)
        closest_sw_height = _sw_peaks_props['peak_heights'][_closest_sw_peak_idx]
    except:
        closest_sw_height = np.nan

    ## Gamma Band
    # Magnitude
    gamma_ave_magni = _gamma_magni_sliced.mean()

    ## EMG
    # Magnitude
    emg_ave_magni = _mep_magni_sliced.mean()

    '''
    ## Check
    plt.plot(_lfp_sliced)
    plt.plot(_ripple_magni_sliced*100)
    plt.plot(_sw_magni_sliced*100*100)
    plt.plot(_gamma_magni_sliced*100*100)
    _wavelet_out = wavelet(_lfp_sliced, SAMP_RATE, f_min=0.1, f_max=500, plot=True)

    print(ripple_peak_magni)
    print(ripple_peak_pos_sec)
    print(ripple_peak_freq_hz)
    print(closest_sw_height)
    print(gamma_ave_magni)
    print(emg_ave_magni)
    '''

    ## Save on memory
    ripple_peaks_magnis.append(ripple_peak_magni)
    ripple_peaks_pos_sec.append(ripple_peak_pos_sec)
    ripple_peaks_freq_hz.append(ripple_peak_freq_hz)
    sw_height.append(closest_sw_height)
    gamma_ave_magnis.append(gamma_ave_magni)
    emg_ave_magnis.append(emg_ave_magni)


ripples['ripple_peaks_pos_sec'] = np.array(ripple_peaks_pos_sec).astype(np.float32)
ripples['ripple_peaks_magnis'] = np.array(ripple_peaks_magnis).astype(np.float32)
ripples['ripple_peaks_freq_hz'] = np.array(ripple_peaks_freq_hz).astype(np.float32)
ripples['sw_height'] = np.array(sw_height).astype(np.float32)
ripples['gamma_ave_magnis'] = np.array(gamma_ave_magnis).astype(np.float32)
ripples['emg_ave_magnis'] = np.array(emg_ave_magnis).astype(np.float32)

ripples['ripple_peaks_magnis_sd'] = ripples['ripple_peaks_magnis'] / SD_RIPPLE_MAGNI
ripples['sw_height_sd'] = ripples['sw_height'] / SD_SW_MAGNI
ripples['gamma_ave_magnis_sd'] = ripples['gamma_ave_magnis'] / SD_GAMMA_MAGNI
ripples['emg_ave_magnis_sd'] = ripples['emg_ave_magnis'] / SD_MEP_MAGNI


## Save
spath = lpath_ripples.replace('.pkl', '_with_prop.pkl')
save_pkl(ripples, spath)


# ## EOF
