## my own packages
import sys
sys.path.append('./')
# sys.path.append('./utils')
import myutils.myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_yolo_191120 import DataloaderFulfiller, load_lfps_rips_sec
sys.path.append('./07_Learning/')
from optimizers import Ranger
from schedulers import cyclical_lr
from apex import amp
sys.path.append('./10_Evaluation/')
from glob_the_best_model_dir import glob_the_last_model_dir
sys.path.append('./11_Models/')
sys.path.append('./11_Models/yolo')
sys.path.append('./11_Models/yolo/utils')
from utils.utils import bounding_ranges_iou
from yolo.models import Darknet
from yolo.data_parallel import DataParallel
# from utils.utils import non_max_suppression_1D
from utils.utils import check_samples_1D, plot_prediction_1D
from skimage import util
from tqdm import tqdm

### others
import argparse
import os
import datetime
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 24
plt.rcParams["figure.figsize"] = (36, 20)

import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
from pprint import pprint
from sklearn.utils import shuffle


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-n", "--npy_fpath", default='../data/01/day1/split/1kHz/tt2-1_fp16.npy', \
                help="The path of the input lfp file (.npy)")
args = ap.parse_args()


## Funcs
def check_with_eyes(ii,
                    lfp,
                    ripples,
                    pred_ripples,
                    obj_thres=0.5,
                    start_sec=10,
                    end_sec=20,
                    samp_rate=1000,
                    savefig=False):

    from collections import OrderedDict

    _pred_ripples = pred_ripples[['start_sec', 'end_sec', 'ripple_prob', 'obj_conf']]\
                                [(start_sec <= pred_ripples['end_sec']) & (pred_ripples['start_sec'] <= end_sec)]\
                                [pred_ripples['obj_conf'] > obj_thres].copy()

    _ripples = ripples[['start_sec', 'end_sec', 'ripple_prob_by_ResNet']]\
                      [(start_sec <= ripples['end_sec']) & (ripples['start_sec'] <= end_sec)]\

    print(_ripples)
    print(_pred_ripples)

    ## Plot
    fig, ax = plt.subplots()
    ymin, ymax = -1000., 1000.
    ax.axis((start_sec, end_sec, ymin, ymax))
    title = 'Obj. Conf. Thres.: {}\n\
             File: {}'.format(obj_thres, args.npy_fpath)
    ax.set_title(title)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Amplitude [uV]')

    start_pts, end_pts = start_sec*samp_rate, end_sec*samp_rate
    x = np.arange(start_sec, end_sec, 1./samp_rate)

    # LFP
    ax.plot(x, lfp[start_pts:end_pts])

    # GT
    # color_dict = {'0':'blue', '1':'red',}
    # label_dict = {'0':'Cleaned-True Ripple', '1':'Cleaned-False Ripple', }
    for _ripple in _ripples.itertuples():
        # color = color_dict[str(int(_ripple.label_cleaned_from_gmm))]
        # label = label_dict[str(int(_ripple.label_cleaned_from_gmm))]
        ax.axvspan(_ripple.start_sec, _ripple.end_sec, alpha=.3*_ripple.ripple_prob_by_ResNet,
                   color='blue', zorder=1000, label='GT Ripple') # GT ripple
        text = 'R: {:.2f}'.format(_ripple.ripple_prob_by_ResNet)
        ax.text((_ripple.start_sec+_ripple.end_sec)/2-0.01, ymax-100, text)

    # color_dict = {'0':'black', '1':'black'}
    for _pred_ripple in _pred_ripples.itertuples():
        # color = color_dict[str(int(_pred_ripple.cls_label))]
        ax.axvspan(_pred_ripple.start_sec, _pred_ripple.end_sec, alpha=0.3*_pred_ripple.ripple_prob,
                   color='black', label='Pred. Ripple', zorder=1000) # Pred. ripple
        text = 'R: {:.2f} \n O: {:.2f}'.format(_pred_ripple.ripple_prob, _pred_ripple.obj_conf)
        ax.text((_pred_ripple.start_sec+_pred_ripple.end_sec)/2-0.01, ymin+100, text)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right')

    if savefig:
        sdir = '/mnt/md0/proj/report/191210/yolo_pred/obj_thres_{}/'.format(obj_thres)
        os.makedirs(sdir, exist_ok=True)
        spath = sdir + '#{}.png'.format(ii)
        fig.savefig(spath)
        print('Saved to: {}'.format(spath))


## Parameters
SAMP_RATE = 1000

# ../data/01/day1/split/1kHz/tt2-1_ripple_pred_with_label_conf_by_yolo.pkl
## Parse File Paths
fpath_lfp = args.npy_fpath # p['fpaths_tes'][0].replace('.npy', '_fp16.npy') # fixme
fpath_pred_ripples = fpath_lfp.replace('_fp16.npy', '_ripple_pred_with_label_conf_by_yolo.pkl')
fpath_ripples = fpath_lfp.replace('_fp16.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm.pkl')


## Load
lfp = np.load(fpath_lfp)
ripples = mf.load_pkl(fpath_ripples)
pred_ripples = mf.load_pkl(fpath_pred_ripples)
'''
plt.hist(pred_ripples['obj_conf'], bins=1000, density=True)
plt.xlabel('Object Confidence')
plt.ylabel('Norm. Num. of Pred. Ripples')
'''

## Create Ripple Prob
# GT ripples
ripples['ripple_prob_by_ResNet'] = 0
indi_true = (ripples['label_cleaned_from_gmm'] == 0)
indi_false = (ripples['label_cleaned_from_gmm'] == 1)
ripples.loc[indi_true, 'ripple_prob_by_ResNet'] = ripples.loc[indi_true, 'prob_pred_by_ResNet']
ripples.loc[indi_false, 'ripple_prob_by_ResNet'] = 1 - ripples.loc[indi_false, 'prob_pred_by_ResNet']
'''
plt.hist(ripples['ripple_prob_by_ResNet'], density=True, bins=1000)
plt.xlabel('Ripple Prob. by ResNet')
plt.ylabel('Norm. Num. of Pred. Ripples')
'''

# Pred ripples
pred_ripples['ripple_prob'] = 0
indi_true = (pred_ripples['cls_label'] == 0)
indi_false = (pred_ripples['cls_label'] == 1)
pred_ripples.loc[indi_true, 'ripple_prob'] = pred_ripples.loc[indi_true, 'cls_conf']
pred_ripples.loc[indi_false, 'ripple_prob'] = 1 - pred_ripples.loc[indi_false, 'cls_conf']
plt.hist(pred_ripples['cls_conf'])
plt.hist(pred_ripples['ripple_prob'], bins=1000)

'''
plt.scatter(pred_ripples['ripple_prob'], pred_ripples['obj_conf'])
plt.xlabel('Ripple Prob.')
plt.ylabel('Obj. Conf')
'''

'''
indi = (pred_ripples['obj_conf'] > 0.90)
plt.hist(pred_ripples['ripple_prob'][indi], bins=1000, density=True)
plt.xlabel('Object Confidence')
plt.ylabel('Norm. Num. of Pred. Ripples')
'''


## Plot the histogram of the prediction score
# plt.hist(pred_ripples['score'], bins=1000)
# plt.ylabel('Number of Predicted Ripples')
# plt.xlabel('Prediction Score')


## Check with Plotting
# pred_ripples = pred_ripples[pred_ripples['cls_label'] == 1]
start_sec, end_sec = 0, 10*60*SAMP_RATE
ripples = ripples[(start_sec < ripples['start_sec']) & (ripples['end_sec'] < end_sec)]
pred_ripples = pred_ripples[(start_sec < pred_ripples['start_sec']) & (pred_ripples['end_sec'] < end_sec)]

obj_thres = 0.5
N_plot = 100
for ii in range(N_plot):
    start_sec = np.random.randint(pred_ripples['start_sec'].iloc[-1])
    end_sec = start_sec + 1
    check_with_eyes(ii,
                    lfp,
                    ripples,
                    pred_ripples,
                    obj_thres=obj_thres,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    savefig=False)


## Calculate AP (Average Precision)
for r in scores:
  pred = pred_ripples[r <= pred_ripples['score']].copy()
  targets = ripples[ripples['label_cleaned_from_gmm' == 1]].copy()
