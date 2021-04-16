import argparse
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
sys.path.append('05_Ripple/rippledetection/')
from core import gaussian_smooth, get_envelope
import numpy as np
import myutils.myfunc as mf
from glob import glob


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath", default='../data/01/day1/split/1kHz/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## Funcs
def parse_fpaths_emg(fpath_lfp):
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


def bandpass(data, lo_hz, hi_hz, fs, order=5):

  def mk_butter_bandpass(order=5):
    from scipy.signal import butter, sosfilt, sosfreqz
    nyq = 0.5 * fs
    low, high = lo_hz/nyq, hi_hz/nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

  def butter_bandpass_filter(data):
    from scipy.signal import butter, sosfilt, sosfreqz
    sos = mk_butter_bandpass()
    y = sosfilt(sos, data)
    return y

  sos = mk_butter_bandpass(order=order)
  y = butter_bandpass_filter(data)

  return y


def calc_band_magnitude(data, samp_rate, lo_hz, hi_hz,
                        devide_by_std=False,
                        minimum_duration=0.15,
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
SAMP_RATE = 1000 # <550 and 2000 doesn't work.
LO_SW, HI_SW = 1, 50
LO_GAMMA, HI_GAMMA = 25, 75
LO_RIPPLE, HI_RIPPLE = 150, 250


## Parse File Paths
fpath_lfp = args.npy_fpath
ldir, fname, ext = mf.split_fpath(fpath_lfp)
fpaths_emg = parse_fpaths_emg(fpath_lfp)

fpath_emg_magni = fpath_lfp.replace('_fp16.npy', '_emg_magni_fp16.npy')
fpath_sw_magni = fpath_lfp.replace('_fp16.npy', '_sw_magni_fp16.npy')
fpath_gamma_magni = fpath_lfp.replace('_fp16.npy', '_gamma_magni_fp16.npy')
fpath_ripple_magni = fpath_lfp.replace('_fp16.npy', '_ripple_magni_fp16.npy')

fpath_emg_magni_sd = fpath_lfp.replace('_fp16.npy', '_emg_magni_sd_fp16.npy')
fpath_sw_magni_sd = fpath_lfp.replace('_fp16.npy', '_sw_magni_sd_fp16.npy')
fpath_gamma_magni_sd = fpath_lfp.replace('_fp16.npy', '_gamma_magni_sd_fp16.npy')
fpath_ripple_magni_sd = fpath_lfp.replace('_fp16.npy', '_ripple_magni_sd_fp16.npy')


## Load
lfp = np.load(fpath_lfp).squeeze().astype(np.float32)[:, np.newaxis]
emg = load_ave_emg(fpaths_emg).astype(np.float32)[:, np.newaxis]
assert len(lfp) == len(emg)


## Magnitudes
emg_magni = calc_band_magnitude(emg, SAMP_RATE, lo_hz=None, hi_hz=None, devide_by_std=False).astype(np.float16)
sw_magni = calc_band_magnitude(lfp, SAMP_RATE, lo_hz=LO_SW, hi_hz=HI_SW, devide_by_std=False).astype(np.float16)
gamma_magni = calc_band_magnitude(lfp, SAMP_RATE, lo_hz=LO_GAMMA, hi_hz=HI_GAMMA, devide_by_std=False).astype(np.float16)
ripple_magni = calc_band_magnitude(lfp, SAMP_RATE, lo_hz=LO_RIPPLE, hi_hz=HI_RIPPLE, devide_by_std=False).astype(np.float16)

emg_magni_sd = calc_band_magnitude(emg, SAMP_RATE, lo_hz=None, hi_hz=None, devide_by_std=True).astype(np.float16)
sw_magni_sd = calc_band_magnitude(lfp, SAMP_RATE, lo_hz=LO_SW, hi_hz=HI_SW, devide_by_std=True).astype(np.float16)
gamma_magni_sd = calc_band_magnitude(lfp, SAMP_RATE, lo_hz=LO_GAMMA, hi_hz=HI_GAMMA, devide_by_std=True).astype(np.float16)
ripple_magni_sd = calc_band_magnitude(lfp, SAMP_RATE, lo_hz=LO_RIPPLE, hi_hz=HI_RIPPLE, devide_by_std=True).astype(np.float16)


## Save
mf.save_npy(emg_magni, fpath_emg_magni)
mf.save_npy(sw_magni, fpath_sw_magni)
mf.save_npy(gamma_magni, fpath_gamma_magni)
mf.save_npy(ripple_magni, fpath_ripple_magni)

mf.save_npy(emg_magni_sd, fpath_emg_magni_sd)
mf.save_npy(sw_magni_sd, fpath_sw_magni_sd)
mf.save_npy(gamma_magni_sd, fpath_gamma_magni_sd)
mf.save_npy(ripple_magni_sd, fpath_ripple_magni_sd)
