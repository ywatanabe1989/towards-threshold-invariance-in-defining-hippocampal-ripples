import argparse
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
sys.path.append('05_Ripple/rippledetection/')
import numpy as np
import myutils.myfunc as mf
from glob import glob
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath", default='../data/01/day1/split/1kHz/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## Parameters
SAMP_RATE = 1000


## Parse File Paths
fpath_lfp = args.npy_fpath
fpath_emg_magni = fpath_lfp.replace('_fp16.npy', '_emg_magni_fp16.npy')
fpath_sw_magni = fpath_lfp.replace('_fp16.npy', '_sw_magni_fp16.npy')
fpath_gamma_magni = fpath_lfp.replace('_fp16.npy', '_gamma_magni_fp16.npy')
fpath_ripple_magni = fpath_lfp.replace('_fp16.npy', '_ripple_magni_fp16.npy')

fpath_emg_magni_sd = fpath_lfp.replace('_fp16.npy', '_emg_magni_sd_fp16.npy')
fpath_sw_magni_sd = fpath_lfp.replace('_fp16.npy', '_sw_magni_sd_fp16.npy')
fpath_gamma_magni_sd = fpath_lfp.replace('_fp16.npy', '_gamma_magni_sd_fp16.npy')
fpath_ripple_magni_sd = fpath_lfp.replace('_fp16.npy', '_ripple_magni_sd_fp16.npy')

# fpath_ripples = fpath_lfp.replace('.npy', '_ripples.pkl')
fpath_ripples = fpath_lfp.replace('.npy', '_ripple_candi_150-250Hz.pkl')


## Load
lfp = np.load(fpath_lfp).squeeze().astype(np.float32).squeeze()
ripples = mf.load_pkl(fpath_ripples)[['start_sec', 'end_sec']]

emg_magni = np.load(fpath_emg_magni).astype(np.float32).squeeze()
sw_magni = np.load(fpath_sw_magni).astype(np.float32).squeeze()
gamma_magni = np.load(fpath_gamma_magni).astype(np.float32).squeeze()
ripple_magni = np.load(fpath_ripple_magni).astype(np.float32).squeeze()

emg_magni_sd = np.load(fpath_emg_magni_sd).astype(np.float32).squeeze()
sw_magni_sd = np.load(fpath_sw_magni_sd).astype(np.float32).squeeze()
gamma_magni_sd = np.load(fpath_gamma_magni_sd).astype(np.float32).squeeze()
ripple_magni_sd = np.load(fpath_ripple_magni_sd).astype(np.float32).squeeze()

SD_EMG_MAGNI = (emg_magni / emg_magni_sd).mean()
SD_SW_MAGNI = (sw_magni / sw_magni_sd).mean()
SD_GAMMA_MAGNI = (gamma_magni / gamma_magni_sd).mean()
SD_RIPPLE_MAGNI = (ripple_magni / ripple_magni_sd).mean()


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
    _emg_magni_sliced = emg_magni[int(_start_pts):int(_end_pts)]
    _gamma_magni_sliced = gamma_magni[int(_start_pts):int(_end_pts)]


    ## Ripple Band
    # Magnitude
    ripple_peak_magni = _ripple_magni_sliced.max()
    # Peak Position
    _ripple_peak_pos_in_slice_pts = _ripple_magni_sliced.argmax()
    ripple_peak_pos_sec =  (_ripple_peak_pos_in_slice_pts + _start_pts) / SAMP_RATE

    # Peak Frequency
    # Wavelet
    _wavelet_out = mf.wavelet(_lfp_sliced, SAMP_RATE, f_min=100, f_max=250, plot=False)
    _col, _row = np.where(_wavelet_out == _wavelet_out.max().max())
    _wavelet_out.iloc[_col, _row]
    ripple_peak_freq_hz = float(np.array(_wavelet_out.index[_col]))

    ## SW Band
    # Height
    try:
        _sw_peaks_pos_pts, _sw_peaks_props = find_peaks(_sw_magni_sliced, height=1e-10)
        _, _closest_sw_peak_idx = mf.take_closest(_sw_peaks_pos_pts, _ripple_peak_pos_in_slice_pts)
        _closest_sw_peak_idx = np.clip(_closest_sw_peak_idx, 0, len(_sw_peaks_props['peak_heights'])-1)
        closest_sw_height = _sw_peaks_props['peak_heights'][_closest_sw_peak_idx]
    except:
        closest_sw_height = np.nan

    ## Gamma Band
    # Magnitude
    gamma_ave_magni = _gamma_magni_sliced.mean()

    ## EMG
    # Magnitude
    emg_ave_magni = _emg_magni_sliced.mean()

    '''
    ## Check
    plt.plot(_lfp_sliced)
    plt.plot(_ripple_magni_sliced*100)
    plt.plot(_sw_magni_sliced*100*100)
    plt.plot(_gamma_magni_sliced*100*100)
    _wavelet_out = mf.wavelet(_lfp_sliced, SAMP_RATE, f_min=0.1, f_max=500, plot=True)

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
ripples['emg_ave_magnis_sd'] = ripples['emg_ave_magnis'] / SD_EMG_MAGNI


## Save
spath = fpath_ripples.replace('.pkl', '_with_prop.pkl')
mf.save_pkl(ripples, spath)


# ## EOF
