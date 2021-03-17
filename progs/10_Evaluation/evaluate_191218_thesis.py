### my own packages
import sys
sys.path.append('./')
sys.path.append('./myutils')
import myfunc as mf
sys.path.append('./06_File_IO')
# from dataloader_191020 import dataloader_fulfiller
from dataloader_191217_thesis import dataloader_fulfiller
sys.path.append('./07_Learning/')
from balance_xentropy_loss import BalanceCrossEntropyLoss
from apex import amp
sys.path.append('./11_Models/')
from model_191015 import Model
sys.path.append('./10_Evaluation/')
from glob_the_best_model_dir import glob_the_best_model_dir

### others
import argparse
import os
import datetime
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
from pprint import pprint
from sklearn.utils import shuffle
from scipy.optimize import curve_fit

plt.rcParams['font.size'] = 24 # 16
plt.rcParams["figure.figsize"] = (36, 20)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nmt", "--n_mouse_tes", default=1, choices=[1,2,3,4,5], type=int, help = "") #  '191022/230633'
ap.add_argument("-save", action='store_true', help = "")
ap.add_argument("-plot", action='store_true', help = "")
args = ap.parse_args()

## Funcs
def sigmoid(x, a, b):
     y = 1. / (1. + np.exp(-b*(x-a)))
     return y


## Load
used_EMG = True


if not used_EMG:
    timestamps = {
    'Test Mouse Number 01':'191219/205715',
    'Test Mouse Number 02':'191219/063042',
    'Test Mouse Number 03':'191219/064927',
    'Test Mouse Number 04':'191219/062310',
    'Test Mouse Number 05':'191219/064946',
    }
    spath_root = '~/Desktop/fig2/'


if used_EMG:
    version = 3
    if version == 1:
        # timestamps = {
        # 'Test Mouse Number 01':'191219/082538',
        # 'Test Mouse Number 02':'191218/232620',
        # 'Test Mouse Number 03':'191218/232632',
        # 'Test Mouse Number 04':'191218/232641',
        # 'Test Mouse Number 05':'191218/232652',
        # }

    if version == 2:
    # # Half GAP, w/o Normalizaion on Input
    # timestamps = {
    # 'Test Mouse Number 01':'191226/051124',
    # 'Test Mouse Number 02':'191226/165627',
    # 'Test Mouse Number 03':'191227/020119',
    # 'Test Mouse Number 04':'191227/111012',
    # 'Test Mouse Number 05':'191227/204739',
    # }

    if version == 3:
    # GAP and w/ Normalization on Input
    timestamps = {
    'Test Mouse Number 01':'191228/200745',
    'Test Mouse Number 02':'191228/110403',
    'Test Mouse Number 03':'191228/110415',
    'Test Mouse Number 04':'191228/110429',
    'Test Mouse Number 05':'191228/110439',
    }

    spath_root = '~/Desktop/fig4/'


ts = timestamps['Test Mouse Number 0{}'.format(args.n_mouse_tes)]

dirpath_root = '../results/' + ts + '/'

dirpaths = mf.natsorted_glob(dirpath_root + 'epoch_*/batch_*/')
lpath = dirpaths[-1]
p = mf.load_pkl(lpath + 'params.pkl')
assert args.n_mouse_tes == int(p['tes_keyword'])

ext = ''
ext = '_new' if version == 2 else ext
ext = '_gap_norminp' if version == 3 else ext

pred_probs = np.load(lpath + 'pred_probs{}.npy'.format(ext))
SDs = np.load(lpath + 'SDs{}.npy'.format(ext))

# indi = (SDs > 0)
# plt.scatter(SDs[indi], pred_isRipple_pred_probs[indi], alpha=0.1)
# plt.scatter(SDs, pred_isRipple_pred_probs, alpha=0.1)

'''
## Check
for i in range(20):
    med = np.median(pred_probs[(i <= SDs) & (SDs < i+1)])
    std = np.std(pred_probs[(i <= SDs) & (SDs < i+1)])
    print('{} < SD <= {}, Median:{:.3f}, Std: {:.3f}'.format(i, i+1, med, std))
'''

## Select outputs of Ripple-including samples
# indi = (SDs > 0)
# _SDs, _pred_probs = SDs[indi], pred_probs[indi]
# indi_sort = np.argsort(_SDs)
# _SDs, _pred_probs = _SDs[indi_sort], _pred_probs[indi_sort]

'''
plt.scatter(_SDs, _pred_probs)
'''

## SD vs Ripple Prob.
SDs
plt.title('Test Mouse#{}\nLabel Cleaning: {}'.format(args.n_mouse_tes, used_EMG))
# Sigmoid Fitting on Original Samples
# popt, pcov = curve_fit(sigmoid, _SDs, _pred_probs, maxfev=10000, bounds=([0., -1e10], [50., 1e10]), method='dogbox')
# popt, pcov = curve_fit(sigmoid, _SDs, _pred_probs, maxfev=10000)
popt, pcov = curve_fit(sigmoid, SDs.squeeze(), pred_probs.squeeze(), maxfev=10000)
x = np.linspace(0, SDs.max(), 1000)
y = sigmoid(x, *popt)
plt.plot(x, y, label='Sigmoid Curve Fitted on Original Samples \ny = 1 / (1 + exp( -b*(x-a)) ) (a:{:.2f}, b:{:.2f})\npcov:\n{}'\
         .format(popt[0], popt[1], pcov))
plt.xlabel('SD')
plt.ylabel('Ripple Prob.')
plt.legend()
# Binning by 0.1 SD and Calculate Mean and STD
bin_SD = 0.1
x_sd = np.arange(0, SDs.max(), bin_SD) + bin_SD/2
means, stds = [], []
flag, i = True, 0
while flag:
    start, end = i*bin_SD, (i+1)*bin_SD
    indi = (start <= SDs) & (SDs < end)
    m, s = pred_probs[indi].mean(), pred_probs[indi].std()
    means.append(m)
    stds.append(s)
    i += 1
    if SDs.max() < end:
        flag = False
plt.errorbar(x_sd, means, yerr=stds, label='bin = 0.1, mean+-std')
xmin, xmax, ymin, ymax = 0.1, 50, -.5, 1.5
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

x = np.arange(xmin, xmax, 0.1)
y0, y1 = np.zeros_like(x), np.ones_like(x)
plt.plot(x, y0, color='black')
plt.plot(x, y1, color='black')
plt.legend()
plt.xscale('log')
# # To csv for SigmaPlot
# df = pd.DataFrame({'x_sd':x_sd,
#                    'means':means,
#                    'stds':stds,
#                    })
# spath_df = '~/Desktop/sigmoid_fitting/{}_label_cleaned_{}.csv'.format(args.n_mouse_tes, used_EMG)
# df.to_csv(spath_df)


## Binary Classification
# Preparation
indi_noRipple, indi_Ripple = (SDs == 0), (7 <= SDs)
pred_probs_noRipple, pred_probs_Ripple = pred_probs[indi_noRipple], pred_probs[indi_Ripple]
y_true = np.hstack([np.zeros(len(pred_probs_noRipple)), np.ones(len(pred_probs_Ripple))]).astype(np.int)
pred_probs_cat = np.concatenate([pred_probs_noRipple, pred_probs_Ripple])
y_pred = (0.5 < pred_probs_cat).astype(np.int)


# Conf Mat
conf_mat = mf.calc_confusion_matrix(y_true, y_pred, labels=['not Ripple', 'Ripple'])
pprint(conf_mat)
# spath_conf_mat = '~/Desktop/conf_mat/{}_label_cleaned_{}.csv'.format(args.n_mouse_tes, used_EMG)
# conf_mat.to_csv(spath_conf_mat)

# Classification Report
cls_rep = mf.report_classification_results(y_true, y_pred, labels=[0, 1])
pprint(cls_rep)
# spath_cls_rep = '~/Desktop/cls_rep/{}_label_cleaned_{}.csv'.format(args.n_mouse_tes, used_EMG)
# cls_rep.to_csv(spath_cls_rep)


# Precision-Recall Curve
precision, recall, thres_pr, pr_auc = mf.calc_precision_recall_curve(y_true, pred_probs_cat, plot=True)

# ROC Curve
fpr, tpr, thres_roc, roc_auc = mf.calc_roc_curve(y_true, pred_probs_cat, plot=True)
# # plt.plot(thres_roc[1:])
