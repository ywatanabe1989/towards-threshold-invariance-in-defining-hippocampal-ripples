import argparse
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
from rippledetection.core import filter_ripple_band, gaussian_smooth
import numpy as np
import myutils.myfunc as mf
from glob import glob
import pandas as pd
# from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['font.size'] = 20
plt.rcParams["figure.figsize"] = (20, 20)
from mpl_toolkits.mplot3d import Axes3D
from plot_ellipsoid import EllipsoidTool
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2
# from scipy.ndimage.filters import gaussian_filter1d
from outliers import smirnov_grubbs as grubbs

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
      rip_sec_parted = rip_sec[rip_sec['label_3d'] == 1]
      label_3d = True
      color = 'blue'
  if plot_true == False:
      rip_sec_parted = rip_sec[rip_sec['label_3d'] == -1]
      label_3d = False
      color = 'red'
  if plot_true == None:
      rip_sec_parted = rip_sec[rip_sec['label_3d'] == 0]
      label_3d = None
      color = 'green'

  n_plot = 0
  while True:
    i_rip = np.random.randint(len(rip_sec_parted))
    start_sec, end_sec = rip_sec_parted.iloc[i_rip]['start_sec'], rip_sec_parted.iloc[i_rip]['end_sec']
    start_pts, end_pts = start_sec*samp_rate, end_sec*samp_rate,
    center_sec = (start_sec + end_sec) / 2
    center_pts = int(center_sec*samp_rate)

    plt_start_pts, plt_end_pts = center_pts - plt_dur_pts, center_pts + plt_dur_pts

    SD = rip_sec_parted.iloc[i_rip]['ripple_peaks_magnis_sd']

    txt = '{} Ripple, SD={:.1f}'.format(label_3d, SD)

    fig, ax = plt.subplots()
    ax.axis((0, plt_end_pts - plt_start_pts, -1500., 1500.))
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

def init():
    # ax.scatter(xx, yy, zz, marker='o', s=20, c="goldenrod", alpha=0.6)
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,


## Parameters
SAMP_RATE = 1000


## Parse File Paths
fpath_lfp = args.npy_fpath
ldir, fname, ext = mf.split_fpath(fpath_lfp)
fpath_ripples = fpath_lfp.replace('.npy', '_ripple_candi_150-250Hz_with_prop.pkl')


## Load
lfp = np.load(fpath_lfp).squeeze().astype(np.float32)[:, np.newaxis]
ripples = mf.load_pkl(fpath_ripples)


## Log-Transformation
ripples['log_duration_ms'] = np.log(ripples['duration_ms'])
ripples['log_emg_ave_magnis_sd'] = np.log(ripples['emg_ave_magnis_sd'])
ripples['log_ripple_peaks_magnis_sd'] = np.log(ripples['ripple_peaks_magnis_sd'])


## Clustering
ftr1, ftr2, ftr3 = 'log_duration_ms', 'log_emg_ave_magnis_sd', 'log_ripple_peaks_magnis_sd'
data = np.array(ripples[[ftr1, ftr2, ftr3]])

gmm = GaussianMixture(n_components=2, covariance_type='full').fit(data)
ripple_clst_idx = np.argmin(gmm.means_[:, 1])
ripple_prob = gmm.predict_proba(data)[:, ripple_clst_idx]


## Labeling
# By Clustering
THRES_CLUSTERING_PROB = 0.5
posi_clust = THRES_CLUSTERING_PROB < ripple_prob
nega_clust = ripple_prob <= (1-THRES_CLUSTERING_PROB)

# By EMG
# THRES_EMG_LWR_PERCENTILE = 5
# idx = int(nega_clust.sum()*(THRES_EMG_LWR_PERCENTILE/100))
thres_log_emg_ave_magnis_sd_lwr = gmm.means_[ripple_clst_idx, 1] # np.sort(ripples['log_emg_ave_magnis_sd'][nega_clust])[idx]
thres_log_emg_ave_magnis_sd_upr = gmm.means_[abs(1-ripple_clst_idx), 1]

none_emg = np.array(thres_log_emg_ave_magnis_sd_lwr <= ripples['log_emg_ave_magnis_sd']) \
         & np.array(ripples['log_emg_ave_magnis_sd'] < thres_log_emg_ave_magnis_sd_upr)

nega_emg = np.array(thres_log_emg_ave_magnis_sd_upr <= ripples['log_emg_ave_magnis_sd'])

# By Ripple Magnitude
THRES_RIPPLE_PEAKS_MAGNIS_SD_LWR = 7
''' fixme
## Lower # fixme
# step = 0.1
# sd_candi = np.arange(0, 16, step)
# resolutions = []
# for sd in sd_candi:
#   ripple_sliced_elipsoid_indi = np.array(np.log(sd+1e-6) < ripples['log_ripple_peaks_magnis_sd']) \
#                               * np.array(ripples['log_ripple_peaks_magnis_sd'] < np.log(sd+1+1e-6))

#   log_emg_ave_magnis_sd_true_sliced = np.sort(ripples[ripple_sliced_elipsoid_indi * posi_clust]['log_emg_ave_magnis_sd'])
#   log_emg_ave_magnis_sd_false_sliced = np.sort(ripples[ripple_sliced_elipsoid_indi * nega_clust]['log_emg_ave_magnis_sd'])

#   try:
#       resolution = log_emg_ave_magnis_sd_false_sliced[int(len(log_emg_ave_magnis_sd_false_sliced)*0.001)] \
#                  - log_emg_ave_magnis_sd_true_sliced[-int(len(log_emg_ave_magnis_sd_true_sliced)*0.001)]
#   except:
#       resolution = None
#   resolutions.append(resolution)

# resolutions = np.array(resolutions, dtype=float)
# resolutions[np.isnan(resolutions)] = 0

# THRES_RESOLUTION_EMG = 1
# resolutions_binary = (THRES_RESOLUTION_EMG < resolutions)
# for i in range(len(resolutions_binary)):
#     if resolutions_binary[i] == False:
#         i_last = i
#     else:
#         pass

# thres_ripple_peaks_magnis_sd_lwr = sd_candi[i_last+1]
'''

## By Duration
THRES_DUR_UPR_PERCENTILE = 5
idx = int(posi_clust.sum()*((100-THRES_DUR_UPR_PERCENTILE)/100))
thres_log_duration_ms_upr = np.sort(ripples['log_duration_ms'][posi_clust])[idx]
thres_duration_ms_upr = np.exp(thres_log_duration_ms_upr)
none_dur = np.array(thres_log_duration_ms_upr < ripples['log_duration_ms'])
# THRES_DURATION_MS_UPR = 400
# THRES_LOG_DURATION_MS_UPR = np.log(THRES_DURATION_MS_UPR)
# none_dur = np.array(THRES_LOG_DURATION_MS_UPR < ripples['log_duration_ms'])
# nega_dur = np.array(THRES_LOG_DURATION_MS < ripples['log_duration_ms'])
# None, Thresholding by constant parameters after clastering
none_clust = ~(posi_clust + nega_clust)
none_ripple_peaks_magnis_sd = np.array( posi_clust
                                        &
                                        (ripples['ripple_peaks_magnis_sd'] < THRES_RIPPLE_PEAKS_MAGNIS_SD_LWR)
                                       )

posi_con = posi_clust ^ none_ripple_peaks_magnis_sd
nega_con = nega_emg # nega_dur # (nega_clust + nega_emg ^ none_dur ^ none_ripple_peaks_magnis_sd)
none_con = none_clust + none_ripple_peaks_magnis_sd + none_emg + none_dur

ripples['label_3d'] = 0 # initialization
ripples.loc[posi_con, 'label_3d'] = 1
ripples.loc[nega_con, 'label_3d'] = -1
ripples.loc[none_con, 'label_3d'] = 0


## Filter out anormal Ripple Magnitude Samples
try:
    alpha = 0.05
    outliers = grubbs.max_test_outliers(np.array(ripples[ripples['label_3d'] == 1]['log_ripple_peaks_magnis_sd']), alpha=0.05)
    isok = np.array([ripples.loc[(ripples['label_3d'] == 1), 'log_ripple_peaks_magnis_sd'].iloc[i] not in outliers
                     for i in range(len(ripples[ripples['label_3d'] == 1]))])
    ripples.loc[(ripples['label_3d'] == 1), 'label_3d'] = isok.astype(np.int)
except:
    pass


## Adjust the number of samples
n_ripples = len(ripples[ripples['label_3d'] == 1])
n_high_emg_noises = len(ripples[ripples['label_3d'] == -1])
if n_ripples < n_high_emg_noises:
  high_emg_noises_sorted = sorted(ripples[ripples['label_3d'] == -1]['log_emg_ave_magnis_sd'])
  _thres_log_emg_ave_magnis_sd_upr = high_emg_noises_sorted[-n_ripples]
  indi = _thres_log_emg_ave_magnis_sd_upr < ripples[ripples['label_3d'] == -1]['log_emg_ave_magnis_sd']
  indi = indi[indi == True]
  ripples.loc[(ripples['label_3d'] == -1), 'label_3d'] = 0
  ripples.loc[indi.index, 'label_3d'] = -1



## Plot 3D scatter
# Prepare sparse Data Frame for visualization
percentage = 10
N = int(len(ripples) * percentage / 100)
indi = np.random.permutation(len(ripples))[:N]
sparse_sd = ripples.iloc[indi]
cls1 = sparse_sd[[ftr1, ftr2, ftr3]][sparse_sd['label_3d'] == 1]
cls2 = sparse_sd[[ftr1, ftr2, ftr3]][sparse_sd['label_3d'] == -1]
cls3 = sparse_sd[[ftr1, ftr2, ftr3]][sparse_sd['label_3d'] == 0]
# Data Points
plot = True
plot_orig = False
plot_clustering = True
plot_ellipsoid = True
save_movie = True
save_png = False
if plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(ftr1)
    ax.set_ylabel(ftr2)
    ax.set_zlabel(ftr3)
    title = 'LFP: {} \n\
             Thres. Peak Ripple Pow.: {}SD\n\
             Thres. Clustering Prob.: {}\n\
             Sparseness: {}%\n\
            '.format(args.npy_fpath, THRES_RIPPLE_PEAKS_MAGNIS_SD_LWR, THRES_CLUSTERING_PROB, percentage)
    plt.title(title)
    ax.axis((2., 8., -3., 3.))
    ax.set_zlim3d(bottom=0., top=3.5)

    if plot_orig:
        ax.scatter(sparse_sd[ftr1], sparse_sd[ftr2], sparse_sd[ftr3])

    # if plot_ellipsoid:
    #     p = 0.0000000000001
    #     ET, radii = EllipsoidTool(), np.array([1., 1., 1.])*np.sqrt(chi2.ppf(1-p, df=3)) # radiouses
    #     ET.plotEllipsoid(gmm.means_[0], radii, gmm.covariances_[0], ax=ax, plotAxes=False) # fixme for scaling
    #     ET.plotEllipsoid(gmm.means_[1], radii, gmm.covariances_[1], ax=ax, plotAxes=False) # fixme for scaling

    if plot_ellipsoid:
        p = 0.01
        r = np.sqrt(chi2.ppf(1-p, df=3))
        ET, radii = EllipsoidTool(), r*np.array([1., 1., 1.]) # radiouses
        ET.plotEllipsoid(gmm.means_[0], radii, gmm.covariances_[0], ax=ax, plotAxes=False) # fixme for scaling
        ET.plotEllipsoid(gmm.means_[1], radii, gmm.covariances_[1], ax=ax, plotAxes=False) # fixme for scaling

    if plot_clustering:
        alpha = 0.3
        ax.scatter(cls1[ftr1], cls1[ftr2], cls1[ftr3],
                   marker='o', label='True Ripple', alpha=alpha)
        ax.scatter(cls2[ftr1], cls2[ftr2], cls2[ftr3],
                   marker='x', label='False Ripple', alpha=alpha)
        ax.scatter(cls3[ftr1], cls3[ftr2], cls3[ftr3],
                   marker='^', label='Not Defined', color='yellowgreen', alpha=alpha)
        plt.legend(loc='upper left')

        if save_png:
            spath_png = (args.npy_fpath).replace('.npy', '_ripple_candi_3d_clst.png')
            plt.savefig(spath_png)
            print("Saved to: {}".format(spath_png))

    if save_movie:
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=360, interval=20, blit=True)
        spath_mp4 = (args.npy_fpath).replace('.npy', '_ripple_candi_3d_clst_191201.mp4')
        print('Saving to: {}'.format(spath_mp4))
        anim.save(spath_mp4, fps=30, extra_args=['-vcodec', 'libx264'])
    else:
      plt.show()




## Plot LFP
do_plot_samples = False
if do_plot_samples:
    plot_samples(lfp.squeeze(), ripples, SAMP_RATE, max_plot=10, plot_true=False, save=False)
    print((ripples['label_3d'] == 1).sum())


## Save
spath = fpath_ripples.replace('.pkl', '_label3d_191201.pkl')
mf.save_pkl(ripples, spath)

## EOF
