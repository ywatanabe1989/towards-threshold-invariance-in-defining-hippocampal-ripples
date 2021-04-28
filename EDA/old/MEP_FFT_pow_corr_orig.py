#!/usr/bin/env python

import argparse
import sys
import numpy as np

sys.path.append('.')
# import myutils.myfunc as mf
import math
from glob import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# plt.rcParams['font.size'] = 14
# w = 4
# plt.rcParams["figure.figsize"] = (w, w*1.618)
# W, H = 6, 1.2
W, H = 15, 10
plt.rcParams["figure.figsize"] = (W, H)
# matplotlib.rcParams['pdf.fonttype'] = 15
matplotlib.rcParams['ps.fonttype'] = 42

import torch.utils.data as utils
import seaborn as sns
# from skimage import util
import skimage
from scipy import fftpack
from tqdm import tqdm
from scipy.stats import zscore
from glob import glob
import natsort
import scipy
from scipy import stats

import utils.general as ug

'''
echo \
01 \
02 \
03 \
04 \
05 \
| xargs -P5 -n1 python 12_Figs/fig_3_EMG_histogram_4_pow_corr.py -nm
'''



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--n_mouse", default='01', choices=['01', '02', '03', '04', '05'], \
                help=" ")
args = ap.parse_args()


## Funcs
# def parse_samp_rate(text):
#   '''
#   parse_samp_rate(p['fpaths_tra'][0]) # 1000
#   '''
#   if text.find('2kHz') >= 0:
#     samp_rate = 2000
#   if text.find('1kHz') >= 0:
#     samp_rate = 1000
#   if text.find('500Hz') >= 0:
#     samp_rate = 500
#   return samp_rate


# def cvt_samp_rate_int2str(**kwargs):
#   '''
#   kwargs = {'samp_rate':500}
#   cvt_samp_rate_int2str(**kwargs) # '500Hz'
#   '''
#   samp_rate = kwargs.get('samp_rate', 1000)
#   samp_rate_estr = '{:e}'.format(samp_rate)
#   e = int(samp_rate_estr[-1])
#   if e == 3:
#     add_str = 'kHz'
#   if e == 2:
#     add_str = '00Hz'
#   samp_rate_str = samp_rate_estr[0] + add_str
#   return samp_rate_str


def cvt_lpath_lfp_2_lpath_rip(lpath_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000}
  test = cvt_lpath_lfp_2_lpath_rip(p['fpaths_tra'][0], **kwargs)
  '''
  samp_rate = parse_samp_rate(lpath_lfp)
  lsamp_str = cvt_samp_rate_int2str(**kwargs)
  lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz')\
                       .replace('_fp16.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm_merged.pkl')
  return lpath_rip


def load_lfp_rip_sec_emg_magni(lpath_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'use_fp16':True}
  lpath_lfp = p['fpaths_tra'][0]
  lfp, rip_sec = load_lfp_rip_sec(p['fpaths_tra'][0], **kwargs)
  '''
  use_fp16 = kwargs.get('use_fp16', True)

  dtype = np.float16 if use_fp16 else np.float32

  lpath_lfp = lpath_lfp.replace('.npy', '_fp16.npy') if use_fp16 else lpath_lfp
  lpath_rip = cvt_lpath_lfp_2_lpath_rip(lpath_lfp, **kwargs)
  lpath_emg_magni = lpath_lfp.replace('_fp16.npy', '_emg_magni_fp16.npy')

  lfp = np.load(lpath_lfp).squeeze().astype(dtype) # 2kHz -> int16, 1kHz, 500Hz -> float32
  rip_sec_df = mf.pkl_load(lpath_rip).astype(float) # Pandas.DataFrame
  emg_magni = np.load(lpath_emg_magni).squeeze().astype(dtype) # 2kHz -> int16, 1kHz, 500Hz -> float32

  return lfp, rip_sec_df, emg_magni


def load_lfps_rips_sec_emg_magnis(lpaths_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'use_fp16':True, 'use_shuffle':True}
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
  '''
  lfps, rips_sec, emg_magnis = [], [], []
  for i in range(len(lpaths_lfp)):
      lpath_lfp = lpaths_lfp[i]
      lfp, rip_sec_df, emg_magni = load_lfp_rip_sec_emg_magni(lpath_lfp, **kwargs)
      lfps.append(lfp)
      rips_sec.append(rip_sec_df)
      emg_magnis.append(emg_magni)

  return lfps, rips_sec, emg_magnis


def extract_unique_emg_magnis(emg_magnis):
    unique_magnis, i_src, i_tgt, is_finished = [], 0, 0, False # Initialization
    unique_magnis.append(emg_magnis[i_src])
    while not is_finished:
        i_tgt += 1

        is_size_matched = (emg_magnis[i_src].size == emg_magnis[i_tgt].size)

        if not is_size_matched:
            unique_magnis.append(emg_magnis[i_tgt])

        if is_size_matched:
            is_duplicated = (emg_magnis[i_src] == emg_magnis[i_tgt]).all()

            if not is_duplicated:
                unique_magnis.append(emg_magnis[i_tgt])

        i_src = i_tgt

        if i_tgt + 1 == len(emg_magnis):
            is_finished = True

    return unique_magnis


def slice_lfp_and_emg_magni(lfp, emg_magni, step=128):
    ## Starting perturbation
    start_perturb_pts = np.random.randint(0, step)
    lfp = lfp[start_perturb_pts:]
    emg_magni = emg_magni[start_perturb_pts:]
    ## Slicing by 512 ms
    slices_lfp = skimage.util.view_as_windows(lfp, window_shape=(step,), step=step)
    slices_emg_magni = util.view_as_windows(emg_magni, window_shape=(step,), step=step)
    return slices_lfp, slices_emg_magni

def slice_lfps_and_emg_magnis(lfps, emg_magnis, step=128):
    out = [slice_lfp_and_emg_magni(lfps[i], emg_magnis[i], step=step) for i in range(len(lfps))]
    lfps_sliced, emg_magnis_sliced = [], []
    for i in range(len(out)):
        lfp_sliced, emg_magni_sliced = out[i]
        lfps_sliced.append(lfp_sliced)
        emg_magnis_sliced.append(emg_magni_sliced)
    lfps_sliced, emg_magnis_sliced = np.vstack(lfps_sliced), np.vstack(emg_magnis_sliced)
    return lfps_sliced, emg_magnis_sliced

def slice_emg_magni(emg_magni, step=128):
    ## Starting perturbation
    start_perturb_pts = np.random.randint(0, step)
    emg_magni = emg_magni[start_perturb_pts:]
    ## Slicing by 512 ms
    slices_emg_magni = util.view_as_windows(emg_magni, window_shape=(step,), step=step)
    return slices_emg_magni

def slice_emg_magnis(emg_magnis, step=128):
    out = [slice_emg_magni(emg_magnis[i], step=step) for i in range(len(emg_magnis))]
    emg_magnis_sliced = []
    for i in range(len(out)):
        emg_magni_sliced = out[i]
        emg_magnis_sliced.append(emg_magni_sliced)
    emg_magnis_sliced = np.vstack(emg_magnis_sliced)
    return emg_magnis_sliced


def calc_fft_powers(x, index):
    X = fftpack.fft(x)
    powers = np.abs(X)[tuple(index)]
    return powers



## Parameters
SAMP_RATE = 1000


## Parse File Paths
LPATHS_NPY_LIST = '../data/1kHz_npy_list.pkl'
LPATHS_EMG_LIST = natsort.natsorted(glob('../data/0?/day?/split/1kHz/tt8*.npy') + glob('../data/0?/day?/split/1kHz/tt3*.npy'))
N_LOAD_ALL = 184
FPATHS_ALL = mf.pkl_load(LPATHS_NPY_LIST)[:N_LOAD_ALL]

FPATHS_MOUSE = []
for f in FPATHS_ALL:
    if 'data/{}'.format(args.n_mouse) in f:
        FPATHS_MOUSE.append(f)

# ttx_txt = 'tt3*' if '../data/02/' in fpath_lfp else 'tt8*'



## Load
# lfps, _, emg_magnis = load_lfps_rips_sec_emg_magnis(FPATHS_ALL)
lfps, _, emg_magnis = load_lfps_rips_sec_emg_magnis(FPATHS_MOUSE)
del _; import gc; gc.collect()
# emg_magnis_z = [zscore(emg_magnis[i].astype(np.float32)).astype(np.float16) for i in range(len(emg_magnis))]
emg_magnis_unique = extract_unique_emg_magnis(emg_magnis)

## Slice lfps and emgs by 512 ms with starting perturbaton
'''
for i in range(len(lfps)):
    print(len(lfps[i]) == len(emg_magnis[i]))
lfp_sliced, emg_magni_sliced = slice_lfp_and_emg_magni(lfps[0], emg_magnis[0])
'''
step = 1024
lfps_sliced, emg_magnis_sliced = slice_lfps_and_emg_magnis(lfps, emg_magnis, step=step)
del lfps, emg_magnis; gc.collect()
# # Choose 1/2 indices for the RAM shortage
# rate = 1/2
# N = len(lfps_sliced)
# rand_indi = np.random.permutation(N)[:int(N*rate)]
# lfps_sliced, emg_magnis_sliced = lfps_sliced[rand_indi], emg_magnis_sliced[rand_indi]
# gc.collect()
mean_emg_magnis_sliced = emg_magnis_sliced.mean(axis=1, keepdims=True)


## EMG Magnitude's Distribution -> Single Gaussian
emg_magnis_unique_sliced = slice_emg_magnis(emg_magnis_unique, step=step)
mean_emg_magnis_unique_sliced = emg_magnis_unique_sliced.mean(axis=1)
# n, bins, patches =
# plt.hist(mean_emg_magnis_unique_sliced.squeeze(), bins=1000)
# plt.xscale('log')
# plt.xlabel('EMG Magnitude [mV]')
# plt.ylabel('Number of {}-ms Samples'.format(step))
# plt.title('n_mouse{}'.format(args.n_mouse))

kurtosis = scipy.stats.kurtosis(mean_emg_magnis_unique_sliced.squeeze().astype(np.float))
skewness = scipy.stats.skew(mean_emg_magnis_unique_sliced.squeeze().astype(np.float))
print("{}, Kurtosis: {}, Skewness:{}".format(args.n_mouse, kurtosis, skewness))


'''
kurtosises = np.array([
217.17702201706078,
143.65302145838288,
166.39092841248026,
191.08142319839752,
185.86044960946992,
])

skewnesses = np.array([
8.872765344956852,
8.035753999520873,
7.6312507644922105,
8.446375941628549,
10.273907771178568,
])
'''











## FFT
# Freq
x = lfps_sliced[0]
f_s = SAMP_RATE
freqs = fftpack.fftfreq(len(x)) * f_s
index = [(0<=freqs)*(freqs<int(f_s/2))]
freqs = freqs[tuple(index)]
# Powers
powers = np.array([calc_fft_powers(lfps_sliced[i], index) for i in tqdm(range(len(lfps_sliced)))])

## Correlation between Frequencies and EMG Magnitude
coeffs = [np.corrcoef(mean_emg_magnis_sliced.squeeze(), powers[:, i].squeeze())[1,0] for i in range(powers.shape[1])]

# coeff_tests = [scipy.stats.pearsonr(mean_emg_magnis_sliced.squeeze().astype(np.float64), powers[:, i].squeeze().astype(np.float64)) for i in range(powers.shape[1])]

## Statistical Test
n = len(coeffs)
ts = np.abs(np.array(coeffs)) * np.sqrt((n - 2) / (1 - np.array(coeffs)**2))
alpha = 0.05
df = n - 2
t_005, t_001 = stats.t.ppf(1-0.05/2, df), stats.t.ppf(1-0.01/2, df)
t_005_mask, t_001_mask = (t_005 < ts), (t_001 < ts)
significant_freq_range_001, significant_freq_range_005 = freqs.copy(), freqs.copy()
significant_freq_range_001[~t_001_mask], significant_freq_range_005[~t_005_mask] = np.nan, np.nan
significant_freq_range_001[t_001_mask], significant_freq_range_005[t_005_mask] = 1, 1
# plt.plot(freqs, coeffs)
# plt.plot(significant_freq_range_005)
# plt.plot(significant_freq_range_001)


'''
plt.plot(freqs, coeffs)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Pearson\'s Correlation Coefficient')
plt.title('Correlation between EMG Magnitude and Powers in Frequency Bands')
'''


## Save
spath_root = '/home/ywatanabe/Desktop/thesis/figs/1/'
# df_mean_emg_magnis_sliced = pd.DataFrame({'mean_emg_magnis_sliced':mean_emg_magnis_sliced.squeeze(),
#                   })
df_mean_emg_magnis_sliced = pd.DataFrame({'mean_emg_magnis_sliced':mean_emg_magnis_unique_sliced.squeeze(),
                  })

df_mean_emg_magnis_sliced.to_csv(spath_root + '{}_mean_emg_magnis_sliced.csv'.format(args.n_mouse))
# Choose 1/2 indices for the RAM shortage
rate = 1/8
# N = len(mean_emg_magnis_sliced)
N = len(mean_emg_magnis_unique_sliced)
rand_indi = np.random.permutation(N)[:int(N*rate)]
# df_mean_emg_magnis_sliced_1per8 = pd.DataFrame({'mean_emg_magnis_sliced1per8':mean_emg_magnis_sliced.squeeze()[rand_indi],
#                   })
df_mean_emg_magnis_sliced_1per8 = pd.DataFrame({'mean_emg_magnis_sliced1per8':mean_emg_magnis_unique_sliced.squeeze()[rand_indi],
                  })
df_mean_emg_magnis_sliced_1per8.to_csv(spath_root + '{}_mean_emg_magnis_sliced1per8.csv'.format(args.n_mouse))


df_coeffs  = pd.DataFrame({'freqs':freqs,
                           'coeffs':np.array(coeffs),
                           'significant_freq_range_005':significant_freq_range_005,
                           'significant_freq_range_001':significant_freq_range_001,
                  }).set_index('freqs')
df_coeffs.to_csv(spath_root + '{}_coeffs.csv'.format(args.n_mouse))
# df_coeffs.plot()

