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
def check_with_eyes(ii, lfp, ripples, pred_ripples,
                    score_thres=0.5, label_3d_thres=0, start_sec=10, end_sec=20, samp_rate=1000, savefig=False):
    from collections import OrderedDict

    _pred_ripples = pred_ripples[['start_sec', 'end_sec', 'duration_ms', 'cls_label', 'score']]\
                                [(start_sec <= pred_ripples['end_sec']) & (pred_ripples['start_sec'] <= end_sec)]\
                                [pred_ripples['score'] > score_thres].copy()

    _ripples = ripples[['start_sec', 'end_sec', 'duration_ms', 'ripple_peaks_magnis_sd', 'label_3d']]\
                      [(start_sec <= ripples['end_sec']) & (ripples['start_sec'] <= end_sec)]\
                      [ripples['label_3d'] >= label_3d_thres].copy()

    print(_ripples)
    print(_pred_ripples)

    ## Plot
    fig, ax = plt.subplots()
    ymin, ymax = -1000., 1000.
    ax.axis((start_sec, end_sec, ymin, ymax))

    title = 'Score Thres.: {}\n\
             File: {}'.format(score_thres, args.npy_fpath)
    ax.set_title(title)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Amplitude [uV]')

    start_pts, end_pts = start_sec*samp_rate, end_sec*samp_rate
    x = np.arange(start_sec, end_sec, 1./samp_rate)
    ax.plot(x, lfp[start_pts:end_pts]) # LFP

    color_dict = {'-1':'red', '0':'green' ,'1':'blue', }
    label_dict = {'-1':'High-EMG-noise Ripple Candi.', '0':'Not Defined Ripple Candi.', '1':'Defined \"True\" Ripples', }
    for _ripple in _ripples.itertuples():
        color = color_dict[str(_ripple.label_3d)]
        label = label_dict[str(_ripple.label_3d)]
        ax.axvspan(_ripple.start_sec, _ripple.end_sec, alpha=.3, color=color_dict[str(_ripple.label_3d)],
                   zorder=1000, label=label_dict[str(_ripple.label_3d)]) # GT ripple

    color_dict = {'0':'black', '1':'magenta'}
    for _pred_ripple in _pred_ripples.itertuples():
        color = color_dict[str(int(_pred_ripple.cls_label))]
        ax.axvspan(_pred_ripple.start_sec, _pred_ripple.end_sec, alpha=0.3*_pred_ripple.score,
                   color=color, label='Pred. Ripples', zorder=1000) # Pred. ripple
        ax.text((_pred_ripple.start_sec+_pred_ripple.end_sec)/2-0.01, ymin+100, '{:.2f}'.format(_pred_ripple.score))

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right')

    if savefig:
        sdir = '/mnt/md0/proj/report/191210/yolo_pred/score_thres_{}/'.format(score_thres)
        os.makedirs(sdir, exist_ok=True)
        spath = sdir + '#{}.png'.format(ii)
        fig.savefig(spath)
        print('Saved to: {}'.format(spath))


# def nms(global_outputs, nms_thres=0.):
#     keep_first = True
#     while global_outputs.size(0):
#         # print(global_outputs.size(0))
#         ii, is_overlap = 0, True
#         while is_overlap:
#             ii += 1
#             if ii < len(global_outputs):
#                 is_overlap = global_outputs[ii, 0] <= global_outputs[0, 1]
#             if ii == len(global_outputs):
#                 is_overlap = False
#         idx_overlap_last =  ii - 1
#         weights = global_outputs[0:idx_overlap_last+1, 2:3]
#         max_obj_conf = weights.max()
#         global_outputs[0, :2] = (weights * global_outputs[0:idx_overlap_last+1, :2]).sum(0) / weights.sum() # Merge
#         global_outputs[0, 2] = max_obj_conf
#         if keep_first:
#             keep_ranges = global_outputs[0].unsqueeze(0)
#             keep_first = False
#         else:
#             keep_ranges = torch.cat([keep_ranges, global_outputs[0].unsqueeze(0)], dim=0)
#         global_outputs = global_outputs[idx_overlap_last+1:]
#     return keep_ranges


# def judge_overlap_existance(nms_outs):
#     starts, ends = nms_outs[:,0], nms_outs[:,1]
#     starts_shift_1, ends_shift_minus1 = nms_outs[1:,0], nms_outs[:-1,1]
#     nms_finished = bool( (starts_shift_1 > ends_shift_minus1).all() )
#     if not nms_finished:
#         indi_false = np.where((starts_shift_1 - ends_shift_minus1) < 0)[0] + 1
#         print(indi_false)
#         print('Rest: {}'.format(len(indi_false)))
#         return False
#     else:
#       return True

# ../data/01/day1/split/1kHz/tt2-1_ripple_pred_selected_binary_by_yolo.pkl
## Parse File Paths
fpath_lfp = args.npy_fpath # p['fpaths_tes'][0].replace('.npy', '_fp16.npy') # fixme
fpath_pred_ripples = fpath_lfp.replace('_fp16.npy', '_ripple_pred_selected_binary_by_yolo.pkl')
fpath_ripples = fpath_lfp.replace('.npy', '_ripple_candi_150-250Hz_with_prop_label3d.pkl')


## Load
lfp = np.load(fpath_lfp)
ripples = mf.load_pkl(fpath_ripples)
pred_ripples = mf.load_pkl(fpath_pred_ripples)


# ## NMS, fixme
# pred_ripples = torch.tensor(np.array(pred_ripples))
# while not judge_overlap_existance(pred_ripples):
#     pred_ripples = nms(pred_ripples, nms_thres=0)
# pred_ripples = pd.DataFrame({'start_sec':pred_ripples[:,0],
#                              'end_sec':pred_ripples[:,1],
#                              'obj_conf':pred_ripples[:,2],
#                              'cls_conf':pred_ripples[:,3],
#                              'cls_label':pred_ripples[:,4],
#                              'ripple_number':np.arange(len(pred_ripples))+1,
#                              })
# pred_ripples.set_index('ripple_number', inplace=True)


# ## Add some properties on predicted ripples
# pred_ripples['duration_ms'] = (pred_ripples['end_sec'] - pred_ripples['start_sec']) * 1000
# pred_ripples['score'] = pred_ripples['obj_conf'] * pred_ripples['cls_conf']


## Plot the histogram of the prediction score
# plt.hist(pred_ripples['score'], bins=1000)
# plt.ylabel('Number of Predicted Ripples')
# plt.xlabel('Prediction Score')


## Check with Plotting
# pred_ripples = pred_ripples[pred_ripples['cls_label'] == 1]
score_thres = 0.5
N_plot = 10
for ii in range(N_plot):
    start_sec = np.random.randint(pred_ripples['start_sec'].iloc[-1])
    end_sec = start_sec + 10
    check_with_eyes(ii, lfp, ripples, pred_ripples,
                    score_thres=score_thres, label_3d_thres=-1, start_sec=start_sec, end_sec=end_sec, savefig=False)


## Calculate AP (Average Precision)
for r in scores:
  pred = pred_ripples[r <= pred_ripples['score']].copy()
  targets = ripples[ripples['label_3d' == 1]].copy()
