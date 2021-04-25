import argparse
import sys
sys.path.append('.')
sys.path.append('05_Ripple/')
import numpy as np
import myutils.myfunc as mf
import pandas as pd


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nm", "--n_mouse", default='05', choices=['01', '02', '03', '04', '05'], \
                help=" ")
args = ap.parse_args()


## Funcs
def load_rip_sec_from_fpath_lfp(lpath_lfp, lpath_rip_last=None):
  if lpath_rip_last is None:
      lpath_rip_last = ''
  lpath_rip = lpath_lfp.replace('.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm{}.pkl'.format(lpath_rip_last))
  rip_sec_df = mf.pkl_load(lpath_rip).astype(float)
  return rip_sec_df


def load_rips_sec(lpaths_lfp, **kwargs):
  rips_sec = []
  for i in range(len(lpaths_lfp)):
      lpath_lfp = lpaths_lfp[i]
      rip_sec_df = load_rip_sec_from_fpath_lfp(lpath_lfp, lpath_rip_last='_merged')
      rips_sec.append(rip_sec_df)
  return rips_sec


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


## Main Loop
lpath_rip_lasts = ['', '_wo_mouse01', '_wo_mouse02', '_wo_mouse03', '_wo_mouse04', '_wo_mouse05']
lpath_rip_lasts.pop(int(args.n_mouse))
keys_to_cat_label = ['label_cleaned_from_gmm' + k for k in lpath_rip_lasts]
keys_to_cat_label_conf = ['prob_pred_by_ResNet' + k for k in lpath_rip_lasts]
for fpath_lfp in FPATHS_MOUSE:
    labels = []
    label_confs = []
    for i, lpath_rip_last in enumerate(lpath_rip_lasts):
        ripple = load_rip_sec_from_fpath_lfp(fpath_lfp, lpath_rip_last=lpath_rip_last)
        label = ripple[keys_to_cat_label[i]]
        label_conf = ripple[keys_to_cat_label_conf[i]]
        labels.append(label)
        label_confs.append(label_conf)

    labels = pd.concat(labels, axis='columns')
    labels_wo = labels[keys_to_cat_label[1:]]

    label_confs = pd.concat(label_confs, axis='columns')
    label_confs_wo = label_confs[keys_to_cat_label_conf[1:]]

    ripples = load_rip_sec_from_fpath_lfp(fpath_lfp, lpath_rip_last=None)
    # Rename within mouse versions
    ripples = ripples.rename(columns={'label_cleaned_from_gmm':'label_cleaned_from_gmm_within_mouse{}'.format(args.n_mouse),
                              'prob_pred_by_ResNet':'prob_pred_by_ResNet_within_mouse{}'.format(args.n_mouse),
                              'noise_idx':'noise_idx_within_mouse{}'.format(args.n_mouse),
                              })

    ripples_merged = pd.concat([ripples, labels_wo, label_confs_wo], axis=1)



    '''
    # plt.scatter(ripples_merged['prob_pred_by_ResNet_wo_mouse01'], ripples_merged['prob_pred_by_ResNet_wo_mouse02'])
    spath = '../data/prob_pred_by_ResNet_corr_mat_about_mouse{}.csv'.format(args.n_mouse)

    ripples_merged[['prob_pred_by_ResNet',
                    'prob_pred_by_ResNet_wo_mouse02',
                    'prob_pred_by_ResNet_wo_mouse03',
                    'prob_pred_by_ResNet_wo_mouse04',
                    'prob_pred_by_ResNet_wo_mouse05']].corr().to_csv(spath)

    mf.save_pkl(pred_prob_corr_mat, )
    '''

    spath = fpath_lfp.replace('.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm_merged.pkl')
    mf.save_pkl(ripples_merged, spath)


## Check
_ripples = load_rips_sec(FPATHS_MOUSE)
ripples = pd.concat(_ripples)


def print_and_save_corr_mat_of_cleaned_label_conf(keys, spath=None):
    keys[0] += '_within_mouse{}'.format(args.n_mouse)

    # sort by mouse num
    n_mice = [int(k[-1]) for k in keys]
    indi = np.argsort(n_mice)
    keys = np.array(keys)[indi]

    corr_mat_label = ripples[keys].corr()
    if spath is not None:
        corr_mat_label.to_csv(spath)
    print(np.array(corr_mat_label))




## Label Conf
spath_label_conf = '../data/corr_mat_label_conf_mouse{}.csv'.format(args.n_mouse)
print_and_save_corr_mat_of_cleaned_label_conf(keys_to_cat_label_conf, spath=spath_label_conf)


def print_and_save_corr_mat_of_cleaned_label(keys, spath=None):
    '''
    keys = keys_to_cat_label
    '''
    keys[0] += '_within_mouse{}'.format(args.n_mouse)

    # sort by mouse num
    n_mice = [int(k[-1]) for k in keys]
    indi = np.argsort(n_mice)
    keys = np.array(keys)[indi]

    corr_mat_label_df = ripples[keys].corr()

    corr_mat_label = np.zeros((len(keys), len(keys)))
    for i, _ in enumerate(keys):
        for j, _ in enumerate(keys):
            corr_mat_label[i, j] = np.array((ripples[keys[i]] == ripples[keys[j]]).mean())

    corr_mat_label_df.loc[:,:] = corr_mat_label

    if spath is not None:
        corr_mat_label_df.to_csv(spath)
    print(np.array(corr_mat_label_df))

## Label
spath_label = '../data/corr_mat_label_mouse{}.csv'.format(args.n_mouse)
print_and_save_corr_mat_of_cleaned_label(keys_to_cat_label, spath=spath_label)


## EOF
