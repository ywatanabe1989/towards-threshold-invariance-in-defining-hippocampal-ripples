import argparse
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
from rippledetection.core import filter_ripple_band, gaussian_smooth
from plot_ellipsoid import EllipsoidTool
sys.path.append('07_Learning/')
from ResNet1D_for_cleaning_labels import CleanLabelResNet1D

import numpy as np
import myutils.myfunc as mf
from glob import glob
import pandas as pd
# from scipy.optimize import curve_fit
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['font.size'] = 20
plt.rcParams["figure.figsize"] = (20, 20)
from mpl_toolkits.mplot3d import Axes3D
import torch.utils.data as utils
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
import cleanlab
from cleanlab.pruning import get_noise_indices
from scipy.stats import chi2


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--n_mouse", default='01', choices=['01', '02', '03', '04', '05'], \
                help=" ")
args = ap.parse_args()



## Funcs
def parse_samp_rate(text):
  '''
  parse_samp_rate(p['fpaths_tra'][0]) # 1000
  '''
  if text.find('2kHz') >= 0:
    samp_rate = 2000
  if text.find('1kHz') >= 0:
    samp_rate = 1000
  if text.find('500Hz') >= 0:
    samp_rate = 500
  return samp_rate


def cvt_samp_rate_int2str(**kwargs):
  '''
  kwargs = {'samp_rate':500}
  cvt_samp_rate_int2str(**kwargs) # '500Hz'
  '''
  samp_rate = kwargs.get('samp_rate', 1000)
  samp_rate_estr = '{:e}'.format(samp_rate)
  e = int(samp_rate_estr[-1])
  if e == 3:
    add_str = 'kHz'
  if e == 2:
    add_str = '00Hz'
  samp_rate_str = samp_rate_estr[0] + add_str
  return samp_rate_str


def cvt_lpath_lfp_2_lpath_rip(lpath_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000}
  test = cvt_lpath_lfp_2_lpath_rip(p['fpaths_tra'][0], **kwargs)
  '''
  samp_rate = parse_samp_rate(lpath_lfp)
  lsamp_str = cvt_samp_rate_int2str(**kwargs)
  lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').replace('_fp16.npy', '_ripple_candi_150-250Hz_with_prop_label_gmm.pkl')
  return lpath_rip

def cvt_lpath_lfp_2_spath_rip(lpath_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000}
  test = cvt_lpath_lfp_2_lpath_rip(p['fpaths_tra'][0], **kwargs)
  '''
  samp_rate = parse_samp_rate(lpath_lfp)
  lsamp_str = cvt_samp_rate_int2str(**kwargs)
  lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').replace('.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm.pkl')
  return lpath_rip


def load_lfp_rip_sec_emg(lpath_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'use_fp16':True}
  lpath_lfp = p['fpaths_tra'][0]
  lfp, rip_sec = load_lfp_rip_sec(p['fpaths_tra'][0], **kwargs)
  '''
  use_fp16 = kwargs.get('use_fp16', True)

  dtype = np.float16 if use_fp16 else np.float32

  lpath_lfp = lpath_lfp.replace('.npy', '_fp16.npy') if use_fp16 else lpath_lfp
  lpath_rip = cvt_lpath_lfp_2_lpath_rip(lpath_lfp, **kwargs)
  lpath_emg = lpath_lfp.replace('_fp16.npy', '_emg_magni_fp16.npy')

  lfp = np.load(lpath_lfp).squeeze().astype(dtype) # 2kHz -> int16, 1kHz, 500Hz -> float32
  rip_sec_df = mf.pkl_load(lpath_rip).astype(float) # Pandas.DataFrame
  emg = np.load(lpath_emg).squeeze().astype(dtype) # 2kHz -> int16, 1kHz, 500Hz -> float32

  return lfp, rip_sec_df, emg


def load_lfps_rips_sec_emgs(lpaths_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'use_fp16':True, 'use_shuffle':True}
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
  '''
  lfps, rips_sec, emgs = [], [], []
  for i in range(len(lpaths_lfp)):
      lpath_lfp = lpaths_lfp[i]
      lfp, rip_sec_df, emg = load_lfp_rip_sec_emg(lpath_lfp, **kwargs)
      lfps.append(lfp)
      rips_sec.append(rip_sec_df)
      emgs.append(emg)

  return lfps, rips_sec, emgs


def pad_sequence(listed_1Darrays, padding_value=0):
    listed_1Darrays = listed_1Darrays.copy()
    dtype = listed_1Darrays[0].dtype

    # get max_len
    max_len = 0
    for i in range(len(listed_1Darrays)):
      max_len = max(max_len, len(listed_1Darrays[i]))

    # padding
    for i in range(len(listed_1Darrays)):
      pad1 = max(0, int((max_len - len(listed_1Darrays[i])) / 2))
      pad2 = max_len - len(listed_1Darrays[i]) - pad1
      listed_1Darrays[i] = np.pad(listed_1Darrays[i], [pad1, pad2], 'constant', constant_values=(padding_value))
    listed_1Darrays = np.array(listed_1Darrays)
    return listed_1Darrays


## Parameters
SAMP_RATE = 1000


## Parse File Paths
LPATHS_NPY_LIST = '../data/1kHz_npy_list.pkl'
N_LOAD_ALL = 184
FPATHS_ALL = mf.pkl_load(LPATHS_NPY_LIST)[:N_LOAD_ALL]
FPATHS_MOUSE = []
for f in FPATHS_ALL:
    if 'data/{}'.format(args.n_mouse) in f:
        FPATHS_MOUSE.append(f)


## Load
lfps, _ripples, emgs = load_lfps_rips_sec_emgs(FPATHS_MOUSE)
lengths = np.array([len(_ripples[i]) for i in range(len(_ripples))])
ripples = pd.concat(_ripples) # Concat


## Cut ripple candidates and make synthetic dataset
cut_lfps, cut_emgs = [], []
max_duration_pts = 400
for i_lfp in range(len(lfps)):
    lfp = lfps[i_lfp]
    ripple = _ripples[i_lfp]
    emg = emgs[i_lfp]
    for i_ripple in range(len(ripple)):
        start_pts = math.floor(ripple.loc[i_ripple+1, 'start_sec']*SAMP_RATE)
        end_pts = math.ceil(ripple.loc[i_ripple+1, 'end_sec']*SAMP_RATE)
        duration_pts = end_pts - start_pts
        center_pts = int((start_pts + end_pts) / 2)
        peak_pos_pts = np.clip(int(ripple.loc[i_ripple+1, 'ripple_peaks_pos_sec']*SAMP_RATE), start_pts, end_pts)
        assert start_pts <= peak_pos_pts & peak_pos_pts <= end_pts

        peak_to_start_pts = peak_pos_pts - start_pts
        peak_to_end_pts = abs(peak_pos_pts - end_pts)

        ## Centerize the peak position
        if peak_to_start_pts <= peak_to_end_pts:
            pad1 = abs(peak_to_end_pts - peak_to_start_pts)
            pad2 = 0
        if peak_to_end_pts < peak_to_start_pts:
            pad1 = 0
            pad2 = abs(peak_to_end_pts - peak_to_start_pts)

        cut_lfp = np.pad(lfp[start_pts:end_pts], [pad1, pad2], 'constant', constant_values=(0))
        cut_emg = np.pad(emg[start_pts:end_pts], [pad1, pad2], 'constant', constant_values=(0))

        if max_duration_pts < len(cut_lfp):
            cut_lfp = cut_lfp[int(len(cut_lfp)/2)-200:int(len(cut_lfp)/2)+200]
            cut_emg = cut_emg[int(len(cut_emg)/2)-200:int(len(cut_emg)/2)+200]

        cut_lfps.append(cut_lfp)
        cut_emgs.append(cut_emg)

assert  len(cut_lfps) == len(ripples)

synthesized_ripples = pad_sequence(cut_lfps) # sythetic data
synthesized_emgs = pad_sequence(cut_emgs) # sythetic data

mf.save_npy(synthesized_ripples, '../data/{}/synthesized_ripples_peak_centered_191231.npy'.format(args.n_mouse))
mf.save_npy(synthesized_emgs, '../data/{}/synthesized_emg_ripple_peak_centered_191231.npy'.format(args.n_mouse))

'''
echo \
01 \
02 \
03 \
04 \
05 \
| xargs -P 5 -n 1 python 05_Ripple/mk_synthesized_ripples_and_emg.py -nm
'''
