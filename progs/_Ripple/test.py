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


## Funcs
def plot_samples(lfp, rip_sec, samp_rate, plt_dur_pts=208, max_plot=1, save=False, plot_true=True):
  if save:
      # import matplotlib
      matplotlib.use('Agg')
      # import matplotlib.pyplot as plt

  if plot_true:
      rip_sec_parted = rip_sec[rip_sec['isTrueRipple'] == 1]
      isTrueRipple = True
      color = 'blue'
  if plot_true == False:
      rip_sec_parted = rip_sec[rip_sec['isTrueRipple'] == -1]
      isTrueRipple = False
      color = 'red'
  if plot_true == None:
      rip_sec_parted = rip_sec[rip_sec['isTrueRipple'] == 0]
      isTrueRipple = None
      color = 'green'

  n_plot = 0
  while True:
    i_rip = np.random.randint(len(rip_sec_parted))
    start_sec, end_sec = rip_sec_parted.iloc[i_rip]['start_sec'], rip_sec_parted.iloc[i_rip]['end_sec']
    start_pts, end_pts = start_sec*samp_rate, end_sec*samp_rate,
    center_sec = (start_sec + end_sec) / 2
    center_pts = int(center_sec*samp_rate)

    plt_start_pts, plt_end_pts = center_pts - plt_dur_pts, center_pts + plt_dur_pts

    SD = rip_sec_parted.iloc[i_rip]['log_ripple_peaks_magnis_sd']

    # if rip_sec_parted.iloc[i_rip]['isTrueRipple'] == 1:
    #     isTrueRipple = True
    # if rip_sec_parted.iloc[i_rip]['isTrueRipple'] == 0:
    #     isTrueRipple = None
    # if rip_sec_parted.iloc[i_rip]['isTrueRipple'] == -1:
    #     isTrueRipple = False

    txt = '{} Ripple, SD={:.1f}'.format(isTrueRipple, SD)

    fig, ax = plt.subplots()
    ax.plot(lfp[plt_start_pts:plt_end_pts])
    ax.axvspan(max(0, start_pts-plt_start_pts),
               min(plt_dur_pts*2, end_pts-plt_start_pts),
               alpha=0.3, color=color, zorder=1000)
    ax.set_title(txt)

    if save:
      spath = '/mnt/md0/proj/report/191126/samples/true/#{}.png'.format(n_plot)
      plt.savefig(spath)

    n_plot += 1
    if n_plot == max_plot:
      break

## Parameters
SAMP_RATE = 1000 # <550 and 2000 doesn't work.


## Parse File Paths
fpath_lfp = args.npy_fpath
fpath_ripples = fpath_lfp.replace('.npy', '_ripples.pkl')


## Load
lfp = np.load(fpath_lfp).squeeze().astype(np.float32).squeeze()
ripples = mf.load_pkl(fpath_ripples)
ripples = ripples.fillna(value={'sw_prominences_sd':0, 'sw_prominences':0})


## Relative peak position
eps = 1e-10
A = (ripples['ripple_peaks_pos_sec'] - ripples['start_sec'])
B = (ripples['end_sec'] - ripples['ripple_peaks_pos_sec'])
ripples['ripple_relat_peak_posi'] = np.log( ( A / (B+eps) ) + eps )


## Log-Transformation
ripples['log_duration_ms'] = np.log(ripples['duration_ms']+1e-10)
ripples['log_ripple_peaks_magnis_sd'] = np.log(ripples['ripple_peaks_magnis_sd']+1e-10)
ripples['log_sw_prominences_sd'] = np.log(ripples['sw_prominences_sd']+1e-10)
ripples['log_gamma_ave_magnis_sd'] = np.log(ripples['gamma_ave_magnis_sd']+1e-10)
ripples['log_emg_ave_magnis_sd'] = np.log(ripples['emg_ave_magnis_sd']+1e-10)


'''
## Select features
ftrs = ['log_duration_ms',
        'ripple_relat_peak_posi',
        'log_ripple_peaks_magnis_sd',
        'ripple_peaks_freq_hz',
        'log_sw_prominences_sd',
        'log_gamma_ave_magnis_sd',
        'log_emg_ave_magnis_sd',
        ]

_ripples = ripples[ftrs]
rename_dict = {
'log_duration_ms':'A',
'ripple_relat_peak_posi':'B',
'log_ripple_peaks_magnis_sd':'C',
'ripple_peaks_freq_hz':'D',
'log_sw_prominences_sd':'E',
'log_gamma_ave_magnis_sd':'F',
'log_emg_ave_magnis_sd':'G',
}

test = ripples.rename(columns=rename_dict)
import seaborn as sns
sns.pairplot(test, diag_kind='kde')
'''
ftrs_gmm = ['log_duration_ms',
            'log_ripple_peaks_magnis_sd',
            'log_emg_ave_magnis_sd',
            ]
            # 'log_gamma_ave_magnis_sd',
            # 'log_sw_prominences_sd',

ripples_gmm = ripples[ftrs_gmm]

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2, covariance_type='full')
data = np.array(ripples_gmm)
idx = np.random.permutation(len(data))[:50000]
gmm = gmm.fit(data[idx])
gmm.score_samples(data)
ripple_prob = gmm.predict_proba(data)[:,1]
cls0_emg, cls1_emg = gmm.means_[0,-1], gmm.means_[1,-1] # Determine classes by EMG power
if cls0_emg < cls1_emg:
  ripple_prob = 1 - ripple_prob

ripples['isTrueRipple'] = ripple_prob > 0.999
plot_samples(lfp, ripples, SAMP_RATE, plt_dur_pts=208, max_plot=10, save=False, plot_true=True)




i, j = np.random.permutation(len(ftrs))[:2]
plt.hist(ripples[ftrs[i]], bins=1000)
plt.title(ftrs[i])

sns.pairplot(ripples[[ftrs[6], ftrs[j]]])



from mpl_toolkits.mplot3d import Axes3D
# ftr1, ftr2, ftr3 = ftrs[5], ftrs[6], ftrs[2]
ftr1, ftr2, ftr3 = ftrs[5], ftrs[6], ftrs[0]
percentage = 10
N = int(len(ripples) * percentage / 100)
indi = np.random.permutation(len(ripples))[:N]
df = ripples[[ftr1, ftr2, ftr3]].iloc[indi]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(ftr1)
ax.set_ylabel(ftr2)
ax.set_zlabel(ftr3)
    # title = 'LFP: {} \n\
    #          Thres. Peak Ripple Pow.: {}SD\n\
    #          Thres. Clustering Prob.: {}\n\
    #          Sparseness: {}%\n\
    #         '.format(args.npy_fpath, thres_peak_ripple_pow_sd, thres_clustering_prob, percentage)
title='title'
plt.title(title)
# ax.axis((1., 8., .2, 9))
# ax.set_zlim3d(bottom=0., top=4.)

alpha = 0.3
ax.scatter(df[ftr1], df[ftr2], df[ftr3],
           marker='o', label='True Ripple', alpha=alpha)
# ax.scatter(cls2[ftr1], cls2[ftr2], cls2[ftr3],
#            marker='x', label='False Ripple', alpha=alpha)
# ax.scatter(cls3[ftr1], cls3[ftr2], cls3[ftr3],
#            marker='^', label='Not Defined', color='yellowgreen', alpha=alpha)
plt.legend(loc='upper left')











































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
# do_shortcut = True
# if not do_shortcut:
sw_prominences = []
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
    # Prominence
    try:
        _sw_peaks_pos_pts, _sw_peaks_props = find_peaks(_sw_magni_sliced, prominence=1e-10)
        _, _closest_sw_peak_idx = mf.take_closest(_sw_peaks_pos_pts, _ripple_peak_pos_in_slice_pts)
        _closest_sw_peak_idx = np.clip(_closest_sw_peak_idx, 0, len(_sw_peaks_props['prominences'])-1)
        closest_sw_prominence = _sw_peaks_props['prominences'][_closest_sw_peak_idx]
    except:
        closest_sw_prominence = np.nan

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
    print(closest_sw_prominence)
    print(gamma_ave_magni)
    print(emg_ave_magni)
    '''

    ## Save on memory
    ripple_peaks_magnis.append(ripple_peak_magni)
    ripple_peaks_pos_sec.append(ripple_peak_pos_sec)
    ripple_peaks_freq_hz.append(ripple_peak_freq_hz)
    sw_prominences.append(closest_sw_prominence)
    gamma_ave_magnis.append(gamma_ave_magni)
    emg_ave_magnis.append(emg_ave_magni)


ripples['ripple_peaks_pos_sec'] = np.array(ripple_peaks_pos_sec).astype(np.float32)
ripples['ripple_peaks_magnis'] = np.array(ripple_peaks_magnis).astype(np.float32)
ripples['ripple_peaks_freq_hz'] = np.array(ripple_peaks_freq_hz).astype(np.float32)
ripples['sw_prominences'] = np.array(sw_prominences).astype(np.float32)
ripples['gamma_ave_magnis'] = np.array(gamma_ave_magnis).astype(np.float32)
ripples['emg_ave_magnis'] = np.array(emg_ave_magnis).astype(np.float32)

ripples['ripple_peaks_magnis_sd'] = ripples['ripple_peaks_magnis'] / SD_RIPPLE_MAGNI
ripples['sw_prominences_sd'] = ripples['sw_prominences'] / SD_SW_MAGNI
ripples['gamma_ave_magnis_sd'] = ripples['gamma_ave_magnis'] / SD_GAMMA_MAGNI
ripples['emg_ave_magnis_sd'] = ripples['emg_ave_magnis'] / SD_EMG_MAGNI


# ## Save
# mf.save_pkl(ripples, fpath_ripples)


# ## EOF
