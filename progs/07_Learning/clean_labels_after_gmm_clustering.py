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
ap.add_argument("-nmt", "--n_mouse_tes", default='01', choices=['01', '02', '03', '04', '05'], \
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
  lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').\
    replace('.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm_wo_mouse{}.pkl'.format(args.n_mouse_tes))
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


def plot_3d_scatter(ripples_mouse,
                    percentage=1,
                    plot=True,
                    plot_orig=False,
                    plot_clustering=True,
                    plot_ellipsoid=True,
                    save_movie=False,
                    save_png=False,
                    ):
    # Prepare sparse Data Frame for visualization
    # percentage = 1
    N = int(len(ripples_mouse) * percentage / 100)
    indi = np.random.permutation(len(ripples_mouse))[:N]
    sparse_sd = ripples_mouse.iloc[indi]
    ftr1, ftr2, ftr3, ftr4 = 'log_duration_ms', 'log_emg_ave_magnis_sd', 'log_ripple_peaks_magnis_sd', 'prob_pred_by_ResNet'
    cls1 = sparse_sd[[ftr1, ftr2, ftr3, ftr4]][sparse_sd['label_cleaned_from_gmm'] == 0]
    cls2 = sparse_sd[[ftr1, ftr2, ftr3, ftr4]][sparse_sd['label_cleaned_from_gmm'] == 1]
    # cls3 = sparse_sd[[ftr1, ftr2, ftr3]][sparse_sd['label_gmm'] == 0]

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
        ax.axis((2., 8., -3., 3.))
        ax.set_zlim3d(bottom=0., top=3.5)

        if plot_orig:
            ax.scatter(sparse_sd[ftr1], sparse_sd[ftr2], sparse_sd[ftr3])

        # if plot_ellipsoid:
        #     p = 0.01
        #     r = np.sqrt(chi2.ppf(1-p, df=3))
        #     ET, radii = EllipsoidTool(), r*np.array([1., 1., 1.]) # radiouses
        #     ET.plotEllipsoid(gmm.means_[0], radii, gmm.covariances_[0], ax=ax, plotAxes=False) # fixme for scaling
        #     ET.plotEllipsoid(gmm.means_[1], radii, gmm.covariances_[1], ax=ax, plotAxes=False) # fixme for scaling

        if plot_clustering:
            alpha = .3
            # for i in range(len(cls1)):
            #     conf = cls1.iloc[i][ftr4]
            #     ax.scatter(cls1.iloc[i][ftr1], cls1.iloc[i][ftr2], cls1.iloc[i][ftr3],
            #                marker='o', label='True Ripple', color='blue', alpha=alpha*conf)

            # for i in range(len(cls2)):
            #     conf = cls2.iloc[i][ftr4]
            #     ax.scatter(cls2.iloc[i][ftr1], cls2.iloc[i][ftr2], cls2.iloc[i][ftr3],
            #                marker='x', label='False Ripple', color='red', alpha=alpha*conf)

            ax.scatter(cls1[ftr1], cls1[ftr2], cls1[ftr3],
                       marker='o', label='True Ripple', alpha=alpha)
            ax.scatter(cls2[ftr1], cls2[ftr2], cls2[ftr3],
                       marker='x', label='False Ripple', alpha=alpha)
            # ax.scatter(cls3[ftr1], cls3[ftr2], cls3[ftr3],
            #            marker='^', label='Not Defined', color='yellowgreen', alpha=alpha)
            plt.legend(loc='upper left')

            if save_png:
                spath_png = (args.npy_fpath).replace('.npy', '_ripple_candi_3d_gmm_only.png')
                plt.savefig(spath_png)
                print("Saved to: {}".format(spath_png))

        if save_movie:
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=360, interval=20, blit=True)
            spath_mp4 = (args.npy_fpath).replace('.npy', '_ripple_candi_3d_clst_gmm_only.mp4')
            print('Saving to: {}'.format(spath_mp4))
            anim.save(spath_mp4, fps=30, extra_args=['-vcodec', 'libx264'])
        else:
          plt.show()



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
LPATHS_NPY_LIST = '../data/1kHz_npy_list.pkl'
N_LOAD_ALL = 184 # fixme
FPATHS_ALL = mf.pkl_load(LPATHS_NPY_LIST)[:N_LOAD_ALL]
FPATHS_MOUSE, _ = mf.split_fpaths(FPATHS_ALL, tes_keyword=args.n_mouse_tes)


## Load
lfps, _ripples = load_lfps_rips_sec(FPATHS_MOUSE)
lengths = np.array([len(_ripples[i]) for i in range(len(_ripples))])
ripples = pd.concat(_ripples) # Concat


## Cut ripple candidates and make synthetic dataset
cut_lfps = []
for i_lfp in range(len(lfps)):
    lfp = lfps[i_lfp]
    ripple = _ripples[i_lfp]
    for i_ripple in range(len(ripple)):
        start_pts, end_pts = int(ripple.loc[i_ripple+1, 'start_sec']*SAMP_RATE), int(ripple.loc[i_ripple+1, 'end_sec']*SAMP_RATE)
        center_pts = int((start_pts + end_pts) / 2)
        length = end_pts - start_pts
        if 400 < length:
            start_pts, end_pts = center_pts - 200, center_pts + 200
        cut_lfp = lfp[start_pts:end_pts]
        cut_lfps.append(cut_lfp)
assert  len(cut_lfps) == len(ripples)
synthesized_ripples = pad_sequence(cut_lfps) # sythetic data
mf.save_npy(synthesized_ripples, '../data/synthesized_ripples_wo_mouse{}.npy'.format(args.n_mouse_tes))
# synthesized_ripples = np.load('../data/{}/synthesized_ripples.npy'.format(args.n_mouse)) # fixme


## Label Conversion
noisy_labels = ripples['label_gmm']
noisy_labels[noisy_labels == 1] = 0
noisy_labels[noisy_labels == -1] = 1
noisy_labels = np.array(noisy_labels).astype(np.int) # target labels


## Prepare for the Cleaning
X_all, y_all = synthesized_ripples[..., np.newaxis], noisy_labels
# N = 10000
# X_all, y_all = X_all[:N], y_all[:N]


## Find Noisy Labels
cl_model = CleanLabelResNet1D(batch_size=1500*4, epochs=2)
cj, psx = cleanlab.latent_estimation.estimate_confident_joint_and_cv_pred_proba(X_all, y_all, clf=cl_model, )
est_py, est_nm, est_inv = cleanlab.latent_estimation.estimate_latent(cj, y_all)
method = ['prune_by_class', 'prune_by_noise_rate', 'both'][2] # method for selecting noise
noise_idx = cleanlab.pruning.get_noise_indices(y_all, psx, est_inv, prune_method=method, confident_joint=cj)
print('Number of estimated errors in training set:', sum(noise_idx))
pred = np.argmax(psx, axis=1)
ordered_noise_idx = np.argsort(np.asarray([psx[i][j] for i,j in enumerate(y_all)])[noise_idx])[::-1]
prob_given = np.asarray([psx[i][j] for i,j in enumerate(y_all)]) # [noise_idx][ordered_noise_idx]
prob_pred = np.asarray([psx[i][j] for i,j in enumerate(pred)]) # [noise_idx][ordered_noise_idx]
# plt.hist(prob_given, density=True)
# plt.hist(prob_pred, density=True)


## Register Cleaned Labels
ripples['prob_pred_by_ResNet_wo_mouse{}'.format(args.n_mouse_tes)] = prob_pred
ripples['noise_idx'] = noise_idx
ripples['label_cleaned_from_gmm_wo_mouse{}'.format(args.n_mouse_tes)] = ripples['label_gmm']
ripples['label_cleaned_from_gmm_wo_mouse{}'.format(args.n_mouse_tes)][noise_idx] = \
      abs(1 - ripples['label_cleaned_from_gmm_wo_mouse{}'.format(args.n_mouse_tes)][noise_idx]) # flip labels
# plot_3d_scatter(ripples) # check with eyes

## Save
for i in range(len(FPATHS_MOUSE)):
    fpath_lfp = FPATHS_MOUSE[i]
    spath_ripple = cvt_lpath_lfp_2_spath_rip(fpath_lfp)
    start, end = lengths[:i].sum(), lengths[:i+1].sum()
    ripple = ripples[start:end]
    mf.save_pkl(ripple, spath_ripple)

'''
men3
export CUDA_VISIBLE_DEVICES=0,1,2,3
python 07_Learning/clean_labels_after_gmm_clustering.py -nmt '01'; python 07_Learning/clean_labels_after_gmm_clustering.py -nmt '02'; python 07_Learning/clean_labels_after_gmm_clustering.py -nmt '03'; python 07_Learning/clean_labels_after_gmm_clustering.py -nmt '04'; python 07_Learning/clean_labels_after_gmm_clustering.py -nmt '05'
'''
