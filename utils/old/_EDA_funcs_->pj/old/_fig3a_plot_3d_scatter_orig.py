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
ap.add_argument("-nm", "--n_mouse", default='02', choices=['01', '02', '03', '04', '05'], \
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
  lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz')\
                       .replace('_fp16.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm_merged.pkl')
  return lpath_rip

# def cvt_lpath_lfp_2_lpath_rip(lpath_lfp, **kwargs):
#   '''
#   kwargs = {'samp_rate':1000}
#   test = cvt_lpath_lfp_2_lpath_rip(p['fpaths_tra'][0], **kwargs)
#   '''
#   samp_rate = parse_samp_rate(lpath_lfp)
#   lsamp_str = cvt_samp_rate_int2str(**kwargs)
#   lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').replace('_fp16.npy', '_ripple_candi_150-250Hz_with_prop_label_gmm.pkl')
#   return lpath_rip


# def cvt_lpath_lfp_2_spath_rip(lpath_lfp, **kwargs):
#   '''
#   kwargs = {'samp_rate':1000}
#   test = cvt_lpath_lfp_2_lpath_rip(p['fpaths_tra'][0], **kwargs)
#   '''
#   samp_rate = parse_samp_rate(lpath_lfp)
#   lsamp_str = cvt_samp_rate_int2str(**kwargs)
#   lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').\
#     replace('.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm_wo_mouse{}.pkl'.format(args.n_mouse_tes))
#   return lpath_rip


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


def pad_sequence(listed_1Darrays, padding_value=0):
    '''
    listed_1Darrays = rips_level_in_slices
    '''
    listed_1Darrays = listed_1Darrays.copy()
    dtype = listed_1Darrays[0].dtype
    # get max_len
    max_len = 0
    for i in range(len(listed_1Darrays)):
      max_len = max(max_len, len(listed_1Darrays[i]))
    # padding
    for i in range(len(listed_1Darrays)):
      # pad = (np.ones(max_len - len(listed_1Darrays[i])) * padding_value).astype(dtype)
      # listed_1Darrays[i] = np.concatenate([listed_1Darrays[i], pad])
      pad1 = int((max_len - len(listed_1Darrays[i])) / 2)
      pad2 = max_len - len(listed_1Darrays[i]) - pad1
      listed_1Darrays[i] = np.pad(listed_1Darrays[i], [pad1, pad2], 'constant', constant_values=(padding_value))
    listed_1Darrays = np.array(listed_1Darrays)
    return listed_1Darrays


def plot_3d_scatter(ripples,
                    cls1_sparse,
                    cls2_sparse,
                    percentage=.5,
                    plot=True,
                    plot_ellipsoid=True,
                    save_movie=False,
                    save_png=False,
                    theta=30,
                    phi=30,
                    ):

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(ftr1)
        ax.set_ylabel(ftr2)
        ax.set_zlabel(ftr3)
        title = 'Mouse #{} \n\
                 Sparsity: {}%\n\
                '.format(args.n_mouse, percentage)
        plt.title(title)
        ax.axis((2.5, 8., -2.5, 3.))
        ax.set_zlim3d(bottom=0., top=3.5)
        ax.view_init(phi, theta)

        alpha = .3
        ax.scatter(cls1_sparse[ftr1], cls1_sparse[ftr2], cls1_sparse[ftr3],
                   marker='o', label='True Ripple', alpha=alpha)
        ax.scatter(cls2_sparse[ftr1], cls2_sparse[ftr2], cls2_sparse[ftr3],
                   marker='o', label='False Ripple', alpha=alpha)
        plt.legend(loc='upper left')

        if plot_ellipsoid:
            from sklearn.mixture import GaussianMixture
            data = ripples[[ftr1, ftr2, ftr3]]
            gmm = GaussianMixture(n_components=2, covariance_type='full').fit(data)
            p = 0.01
            r = np.sqrt(chi2.ppf(1-p, df=3))
            ET, radii = EllipsoidTool(), r*np.array([1., 1., 1.]) # radiouses
            ET.plotEllipsoid(gmm.means_[0], radii, gmm.covariances_[0], ax=ax, plotAxes=False, cageAlpha=0.1) # fixme for scaling
            ET.plotEllipsoid(gmm.means_[1], radii, gmm.covariances_[1], ax=ax, plotAxes=False, cageAlpha=0.1) # fixme for scaling


        if save_png:
            # spath_png = (args.npy_fpath).replace('.npy', '_ripple_candi_3d_gmm_only.png')
            plt.savefig(spath_png)
            print("Saved to: {}".format(spath_png))

        if save_movie:
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=360, interval=20, blit=True)
            # spath_mp4 = (args.npy_fpath).replace('.npy', '_ripple_candi_3d_clst_gmm_only.mp4')
            print('Saving to: {}'.format(spath_mp4))
            anim.save(spath_mp4, fps=30, extra_args=['-vcodec', 'libx264'])
        else:
          plt.show()




# def plot_samples(lfp, rip_sec, samp_rate, plt_dur_pts=208, max_plot=1, save=False, plot_true=True):
#   if save:
#       # import matplotlib
#       matplotlib.use('Agg')
#       # import matplotlib.pyplot as plt

#   if plot_true:
#       rip_sec_parted = rip_sec[rip_sec['label_3d'] == 1]
#       label_3d = True
#       color = 'blue'
#   if plot_true == False:
#       rip_sec_parted = rip_sec[rip_sec['label_3d'] == -1]
#       label_3d = False
#       color = 'red'
#   if plot_true == None:
#       rip_sec_parted = rip_sec[rip_sec['label_3d'] == 0]
#       label_3d = None
#       color = 'green'

#   n_plot = 0
#   while True:
#     i_rip = np.random.randint(len(rip_sec_parted))
#     start_sec, end_sec = rip_sec_parted.iloc[i_rip]['start_sec'], rip_sec_parted.iloc[i_rip]['end_sec']
#     start_pts, end_pts = start_sec*samp_rate, end_sec*samp_rate,
#     center_sec = (start_sec + end_sec) / 2
#     center_pts = int(center_sec*samp_rate)

#     plt_start_pts, plt_end_pts = center_pts - plt_dur_pts, center_pts + plt_dur_pts

#     SD = rip_sec_parted.iloc[i_rip]['ripple_peaks_magnis_sd']

#     txt = '{} Ripple, SD={:.1f}'.format(label_3d, SD)

#     fig, ax = plt.subplots()
#     ax.axis((0, plt_end_pts - plt_start_pts, -1500., 1500.))
#     ax.plot(lfp[plt_start_pts:plt_end_pts])
#     ax.axvspan(max(0, start_pts-plt_start_pts),
#                min(plt_dur_pts*2, end_pts-plt_start_pts),
#                alpha=0.3, color=color, zorder=1000)
#     ax.set_title(txt)


#     if save:
#       spath = '/mnt/md0/proj/report/191126/samples/true/#{}.png'.format(n_plot)
#       plt.savefig(spath)

#     n_plot += 1
#     if n_plot == max_plot:
#       break

# def init():
#     # ax.scatter(xx, yy, zz, marker='o', s=20, c="goldenrod", alpha=0.6)
#     return fig,

# def animate(i):
#     ax.view_init(elev=10., azim=i)
#     return fig,


## Parameters
SAMP_RATE = 1000


## Parse File Paths
LPATHS_NPY_LIST = '../data/1kHz_npy_list.pkl'
N_LOAD_ALL = 184 # fixme
FPATHS_ALL = mf.pkl_load(LPATHS_NPY_LIST)[:N_LOAD_ALL]
FPATHS_MOUSE = []
for f in FPATHS_ALL:
    if 'data/{}'.format(args.n_mouse) in f:
        FPATHS_MOUSE.append(f)

## Load
lfps, _ripples = load_lfps_rips_sec(FPATHS_MOUSE)
lengths = np.array([len(_ripples[i]) for i in range(len(_ripples))])
ripples = pd.concat(_ripples) # Concat
ripples = ripples[ripples['ripple_peaks_magnis_sd'] >= 1]


# Fixme, the index of sparse
# Prepare sparse Data Frame for visualization
# percentage = .03
percentage = .1#, fig3a
N = int(len(ripples) * percentage / 100)
indi_sparse = np.random.permutation(len(ripples))[:N]
sparse_sd = ripples.iloc[indi_sparse]

ftr1, ftr2, ftr3 = 'log_duration_ms', 'log_emg_ave_magnis_sd', 'log_ripple_peaks_magnis_sd'

# label_name_cleaned = 'label_cleaned_from_gmm_within_mouse{}'.format(args.n_mouse)
label_name_cleaned = 'label_cleaned_from_gmm_wo_mouse01'
indi_t_cleaned = sparse_sd[label_name_cleaned] == 0
indi_f_cleaned = sparse_sd[label_name_cleaned] == 1
t_cleaned = sparse_sd[[ftr1, ftr2, ftr3]][indi_t_cleaned]
f_cleaned = sparse_sd[[ftr1, ftr2, ftr3]][indi_f_cleaned]

label_name_gmm = 'label_gmm'
indi_t_gmm = sparse_sd[label_name_gmm] == 0
indi_f_gmm = sparse_sd[label_name_gmm] == 1
t_gmm = sparse_sd[[ftr1, ftr2, ftr3]][indi_t_gmm]
f_gmm = sparse_sd[[ftr1, ftr2, ftr3]][indi_f_gmm]

theta, phi = 165, 3
# plot_3d_scatter(ripples, t_gmm, f_gmm, plot_ellipsoid=False, theta=theta, phi=phi, percentage=percentage)
# plot_3d_scatter(ripples, t_cleaned, f_cleaned, plot_ellipsoid=False, theta=theta, phi=phi, percentage=percentage)

indi_t2t = indi_t_gmm & indi_t_cleaned
indi_f2f = indi_f_gmm & indi_f_cleaned
indi_f2t = indi_f_gmm & indi_t_cleaned
indi_t2f = indi_t_gmm & indi_f_cleaned

t2f = sparse_sd[[ftr1, ftr2, ftr3]][indi_t2f]
f2t = sparse_sd[[ftr1, ftr2, ftr3]][indi_f2t]
plot_3d_scatter(ripples, f2t, t2f, plot_ellipsoid=True, theta=theta, phi=phi, percentage=percentage)
# ## Save
# spath_root = '~/Desktop/fig3a/'
# pos_gmm.to_csv(spath_root + 'pos_gmm.csv')
# neg_gmm.to_csv(spath_root + 'neg_gmm.csv')
# pos_cleaned.to_csv(spath_root + 'pos_cleaned.csv')
# neg_cleaned.to_csv(spath_root + 'neg_cleaned.csv')
