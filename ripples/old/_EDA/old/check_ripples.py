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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
w = 8
plt.rcParams["figure.figsize"] = (w, w*1.618)
matplotlib.rcParams['pdf.fonttype'] = 15
matplotlib.rcParams['ps.fonttype'] = 42
from mpl_toolkits.mplot3d import Axes3D
import torch.utils.data as utils
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
import cleanlab
from cleanlab.pruning import get_noise_indices
from scipy.stats import chi2
import seaborn as sns


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

# def cvt_lpath_lfp_2_lpath_rip(lpath_lfp, **kwargs):
#   '''
#   kwargs = {'samp_rate':1000}
#   test = cvt_lpath_lfp_2_lpath_rip(p['fpaths_tra'][0], **kwargs)
#   '''
#   samp_rate = parse_samp_rate(lpath_lfp)
#   lsamp_str = cvt_samp_rate_int2str(**kwargs)
#   lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').replace('_fp16.npy', '_ripple_candi_150-250Hz_with_prop_label_gmm.pkl')
#   return lpath_rip


def cvt_lpath_lfp_2_lpath_rip(lpath_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000}
  test = cvt_lpath_lfp_2_lpath_rip(p['fpaths_tra'][0], **kwargs)
  '''
  samp_rate = parse_samp_rate(lpath_lfp)
  lsamp_str = cvt_samp_rate_int2str(**kwargs)
  lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').replace('_fp16.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm.pkl')
  return lpath_rip


def load_lfp_rip_sec(lpath_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'use_fp16':True}
  lpath_lfp = p['fpaths_tra'][0]
  lfp, rip_sec = load_lfp_rip_sec(p['fpaths_tra'][0], **kwargs)
  '''
  use_fp16 = kwargs.get('use_fp16', True)

  dtype = np.float16 if use_fp16 else np.float32

  lpath_lfp = lpath_lfp.replace('.npy', '_fp16.npy') if use_fp16 else lpath_lfp
  lpath_rip = cvt_lpath_lfp_2_lpath_rip(lpath_lfp, **kwargs)

  lfp = np.load(lpath_lfp).squeeze().astype(dtype) # 2kHz -> int16, 1kHz, 500Hz -> float32
  rip_sec_df = mf.pkl_load(lpath_rip).astype(float) # Pandas.DataFrame

  return lfp, rip_sec_df


def load_lfps_rips_sec(lpaths_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'use_fp16':True, 'use_shuffle':True}
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
  '''
  lfps, rips_sec = [], []
  for i in range(len(lpaths_lfp)):
      lpath_lfp = lpaths_lfp[i]
      lfp, rip_sec_df = load_lfp_rip_sec(lpath_lfp, **kwargs)
      lfps.append(lfp)
      rips_sec.append(rip_sec_df)

  if kwargs.get('use_shuffle', False):
    lfps, rips_sec = shuffle(lfps, rips_sec) # 1st shuffle

  return lfps, rips_sec


def plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t, xftr,
                 xlabel=None, xscale=None, alpha=0.01, sparcity=100, flipped_only=False):
    import math
    i_t2t = np.random.permutation(len(ripples_t2t))[:(math.ceil(sparcity/100 * len(ripples_t2t)))]
    i_f2f = np.random.permutation(len(ripples_f2f))[:(math.ceil(sparcity/100 * len(ripples_f2f)))]
    i_f2t = np.random.permutation(len(ripples_f2t))[:(math.ceil(sparcity/100 * len(ripples_f2t)))]
    i_t2f = np.random.permutation(len(ripples_t2f))[:(math.ceil(sparcity/100 * len(ripples_t2f)))]

    ripples_t2t = ripples_t2t.iloc[i_t2t]
    ripples_f2f = ripples_f2f.iloc[i_f2f]
    ripples_f2t = ripples_f2t.iloc[i_f2t]
    ripples_t2f = ripples_t2f.iloc[i_t2f]

    xlabel = xftr if (xlabel == None) else xlabel
    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)

    plt.ylim((.5, 1.))

    ylabel = 'Pred. Prob. by ResNet'
    xlabel_ax0 = 'Cleaned-True Ripple Samples' if not flipped_only else 'Flipped-True Ripple Samples'
    xlabel_ax1 = 'Cleaned-False Ripple Samples' if not flipped_only else 'Flipped-False Ripple Samples'
    ax[0].set_title(xlabel_ax0)
    ax[0].set_ylabel(ylabel)
    ax[1].set_title(xlabel_ax1)
    ax[1].set_ylabel(ylabel)

    color_t2t = 'black' # 'blue'
    color_f2f = 'black' # 'red'
    color_t2f = 'black' # 'green'
    color_f2t = 'black' # 'green'

    base_s = 30

    if not flipped_only:
        ax[0].scatter(ripples_t2t[xftr], ripples_t2t['prob_pred_by_ResNet'],
                      alpha=alpha, color=color_t2t, marker='x', label='True to True', s=base_s)

    ax[0].scatter(ripples_f2t[xftr], ripples_f2t['prob_pred_by_ResNet'],
                  alpha=alpha, color=color_f2t, marker='o', label='False to True', facecolors='none', s=base_s*2)
    ax[0].legend()

    if not flipped_only:
        ax[1].scatter(ripples_f2f[xftr], ripples_f2f['prob_pred_by_ResNet'],
                      alpha=alpha, color=color_f2f, marker='x', label='False to False', s=base_s)

    ax[1].scatter(ripples_t2f[xftr], ripples_t2f['prob_pred_by_ResNet'],
                  alpha=alpha, color=color_t2f, marker='o', label='True to False', facecolors='none', s=base_s*2)
    ax[1].legend()

    plt.legend()

    if xscale != None:
        plt.xscale(xscale)
    plt.xlabel(xlabel)




def plot_mean_waves(syn_ripples_true, syn_ripples_false, ax0title=None, ax1title=None):
    mean_true, std_true = syn_ripples_true.mean(axis=0), syn_ripples_true.std(axis=0)
    mean_false, std_false = syn_ripples_false.mean(axis=0), syn_ripples_false.std(axis=0)

    fig, ax = plt.subplots(1,2, sharey=True)
    ax[0].axis((-200, 200, -400., 800.))

    ylabel = 'Amplitude [uV]'
    xlabel = 'Time from Ripple Peak [ms]'
    x = np.arange(-200, 200)

    alpha=0.3
    ax[0].plot(x, mean_true, color='black')
    ax[0].fill_between(x, mean_true - std_true, mean_true + std_true, color='blue', alpha=alpha)
    # ax[0].set_title('Cleaned-True Ripple Samples (Mean +- Std)')
    ax[0].set_title(ax0title)
    ax[0].set_ylabel(ylabel)
    ax[0].set_xlabel(xlabel)

    ax[1].plot(x, mean_false, color='black')
    ax[1].fill_between(x, mean_false - std_false, mean_false + std_false, color='red', alpha=alpha)
    # ax[1].set_title('Cleaned-False Ripple Samples (Mean +- Std)')
    ax[1].set_title(ax1title)
    ax[1].set_xlabel(xlabel)


def calc_dtw(srcs, tgt):
    import fastdtw
    assert len(tgt.shape) == 1
    distances = []
    for i in range(len(srcs)):
        src = srcs[i]
        distance, path = fastdtw.dtw(tgt, src)
        n_elements = (src != 0).sum()
        distances.append(distance/n_elements)
    distances = np.array(distances)
    return distances

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
lfps, _ripples = load_lfps_rips_sec(FPATHS_MOUSE)
lengths = np.array([len(_ripples[i]) for i in range(len(_ripples))])
ripples = pd.concat(_ripples) # Concat



# ripples = mf.load_pkl('../data/01/day1/split/1kHz/tt2-1_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm.pkl')
indi_true = (ripples['label_cleaned_from_gmm'] == 0)
indi_false = (ripples['label_cleaned_from_gmm'] == 1)
## Cvt to Ripple Prob
ripples['ripple_prob_by_ResNet'] = 0
ripples.loc[indi_true, 'ripple_prob_by_ResNet'] = ripples.loc[indi_true, 'prob_pred_by_ResNet']
ripples.loc[indi_false, 'ripple_prob_by_ResNet'] = 1 - ripples.loc[indi_false, 'prob_pred_by_ResNet']

'''
plt.hist(ripples['ripple_prob_by_ResNet'], bins=1000)
plt.title('Mouse #{}'.format(args.n_mouse))
plt.xlabel('Ripple Prob. Pred. by ResNet')
plt.ylabel('Num. of Samples')
'''


indi_flipped = ripples['noise_idx'].astype(np.bool)
ripples_true = ripples[indi_true]
ripples_false = ripples[indi_false]

indi_t2t = ~indi_flipped & indi_true
indi_f2f = ~indi_flipped & indi_false
indi_f2t = indi_flipped & indi_true
indi_t2f = indi_flipped & indi_false

ripples_t2t = ripples[indi_t2t]
ripples_f2f = ripples[indi_f2f]
ripples_t2f = ripples[indi_t2f]
ripples_f2t = ripples[indi_f2t]




## "SD" vs prob_pred_by_ResNet
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'ripple_peaks_magnis_sd', xlabel='Ripple Peak Magnitude / SD [uV]', xscale='log', sparcity=50)
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'ripple_peaks_magnis_sd', xlabel='Ripple Peak Magnitude / SD [uV]',
             xscale='log', sparcity=50, flipped_only=True)

## Duration vs prob_pred_by_ResNet
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'duration_ms', xlabel='Duration [ms]', xscale='log')
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'duration_ms', xlabel='Duration [ms]', xscale='log', flipped_only=True)

## Freq vs prob_pred_by_ResNet
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'ripple_peaks_freq_hz', xlabel='Ripple Band Peak Frequency [Hz]')
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'ripple_peaks_freq_hz', xlabel='Ripple Band Peak Frequency [Hz]', flipped_only=True)

## SW height vs prob_pred_by_ResNet
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'sw_height', xlabel='Sharp Wave Height [uV]', xscale='log')
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'sw_height', xlabel='Sharp Wave Height [uV]', xscale='log', flipped_only=True)

## Gamma Magnitude vs prob_pred_by_ResNet
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'gamma_ave_magnis', xlabel='Gamma Average Magnitude / SD [uV]', xscale='log')
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'gamma_ave_magnis', xlabel='Gamma Average Magnitude / SD [uV]', xscale='log', flipped_only=True)

## EMG Average Magnitude vs prob_pred_by_ResNet
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'emg_ave_magnis', xlabel='EMG Average Magnitude / SD [uV]', xscale='log', sparcity=5, alpha=.5)
plot_scatter(ripples_t2t, ripples_f2f, ripples_t2f, ripples_f2t,
             'emg_ave_magnis', xlabel='EMG Average Magnitude / SD [uV]', xscale='log', sparcity=1, flipped_only=True, alpha=1)



# ## Noise idx vs GMM Weak Label
# cj = np.load('../data/01/cj.npy')
# psx = np.load('../data/01/psx.npy')
# est_py = np.load('../data/01/est_py.npy')
# est_nm = np.load('../data/01/est_nm.npy')
# est_inv = np.load('../data/01/est_inv.npy')

# synthesized_ripples = np.load('../data/01/synthesized_ripples.npy')
synthesized_ripples = np.load('../data/01/synthesized_ripples_peak_centered.npy')

## Mean Images
syn_true = synthesized_ripples[indi_true].astype(np.float128)
syn_false = synthesized_ripples[indi_false].astype(np.float128)
n_true, n_false = indi_true.sum(), indi_false.sum()
ax0title = 'Cleaned-True Ripple Samples \n (Mean +- Std, n={})'.format(n_true)
ax1title = 'Cleaned-False Ripple Samples \n (Mean +- Std, n={})'.format(n_false)
plot_mean_waves(syn_true, syn_false, ax0title=ax0title, ax1title=ax1title)

syn_f2t = synthesized_ripples[indi_f2t].astype(np.float128)
syn_t2f = synthesized_ripples[indi_t2f].astype(np.float128)
n_f2t, n_t2f = indi_f2t.sum(), indi_t2f.sum()
ax0title = 'Flipped from False to True \n (Mean +- Std, n={})'.format(n_f2t)
ax1title = 'Flipped from True to False \n (Mean +- Std, n={})'.format(n_t2f)
plot_mean_waves(syn_f2t, syn_t2f, ax0title=ax0title, ax1title=ax1title)


## Calc Dinamic Time Warping
syn_true_mean = syn_true.mean(axis=0)
syn_false_mean = syn_false.mean(axis=0)

dist_f2t_to_true_mean = calc_dtw(syn_f2t, syn_true_mean)
dist_f2t_to_false_mean = calc_dtw(syn_f2t, syn_false_mean)
dist_t2f_to_true_mean = calc_dtw(syn_t2f, syn_true_mean)
dist_t2f_to_false_mean = calc_dtw(syn_t2f, syn_false_mean)


# plt.hist(dist_f2t_to_true_mean, bins=1000, range=[0, 30000], label='f2t -> true', density=True)
# plt.hist(dist_f2t_to_false_mean, bins=1000, range=[0, 30000], label='f2t -> false', density=True)
# plt.hist(dist_t2f_to_true_mean, bins=1000, range=[0, 30000], label='t2f -> true', density=True)
# plt.hist(dist_t2f_to_false_mean, bins=1000, range=[0, 30000], label='t2f -> false', density=True)
# plt.legend()


# sns.distplot(dist_f2t_to_true_mean, bins=1000, label='f2t -> true', kde=True, norm_hist=True)
# sns.distplot(dist_f2t_to_false_mean, bins=1000, label='f2t -> false', kde=True, norm_hist=True)
# sns.distplot(dist_t2f_to_true_mean, bins=1000, label='t2f -> true', kde=True, norm_hist=True)
# sns.distplot(dist_t2f_to_false_mean, bins=1000, label='t2f -> false', kde=True, norm_hist=True)


sns.kdeplot(dist_f2t_to_true_mean, label='f2t -> true')
sns.kdeplot(dist_f2t_to_false_mean, label='f2t -> false')
sns.kdeplot(dist_t2f_to_true_mean, label='t2f -> true')
sns.kdeplot(dist_t2f_to_false_mean, label='t2f -> false')
plt.xlim(0, 300)


print(dist_f2t_to_true_mean.mean())
print(dist_f2t_to_false_mean.mean())
print(dist_t2f_to_true_mean.mean())
print(dist_t2f_to_false_mean.mean())
