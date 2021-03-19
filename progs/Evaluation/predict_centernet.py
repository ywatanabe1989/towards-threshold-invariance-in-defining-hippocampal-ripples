## my own packages
import sys
sys.path.append('./')
# sys.path.append('./utils')
import myutils.myfunc as mf
sys.path.append('./06_File_IO')
# from dataloader_yolo_191120 import DataloaderFulfiller, load_lfps_rips_sec
from dataloader_centernet import DataloaderFulfiller, load_lfps_rips_sec
sys.path.append('./07_Learning/')
# from CenterNet import CenterNet1D
# from CenterNet_wo_padding import CenterNet1D_wo_bp as CenterNet1D
from optimizers import Ranger
from schedulers import cyclical_lr
from apex import amp
sys.path.append('./10_Evaluation/')
from glob_the_best_model_dir import glob_the_last_model_dir
sys.path.append('./11_Models/')
sys.path.append('./11_Models/feature_extractors/')
from CenterNet_after_SincNet import CenterNet1D_after_SincNet as CenterNet1D
sys.path.append('./11_Models/yolo')
sys.path.append('./11_Models/yolo/utils')
from utils.utils import bounding_ranges_iou
from yolo.models import Darknet
from yolo.data_parallel import DataParallel
from utils.utils import non_max_suppression_1D
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
import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
from pprint import pprint
from sklearn.utils import shuffle


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nmt", "--n_mouse_tes", default="01", choices=["01", "02", "03", "04", "05"], help = " ")
# 12: Fast, 176: w/o 05/day5, 186: Full
ap.add_argument("--model_def", default='./11_Models/yolo/config/yolov3.cfg', help=" ")
ap.add_argument("-msl", "--max_seq_len", default=416, choices=[52, 104, 384, 416], type=int, help=" ") # input_len
args = ap.parse_args()


## Funcs
def load_params(dirpath, plot=False):
  timer = mf.time_tracker()
  print('Data directory: {}'.format(dirpath))
  d = mf.pkl_load(dirpath + 'data.pkl')
  p = mf.pkl_load(dirpath + 'params.pkl')
  m = mf.pkl_load(dirpath + 'model.pkl')
  p['bs_tes'] = 1024 * 10
  weight_path = dirpath + 'weight.pth'
  return d, p, m, timer, weight_path, dirpath


def init_NN(p, m):
  print('Initializing NN')

  model = CenterNet1D().to(m['device'])

  optimizer = Ranger(model.parameters(), lr=m['init_lr'], eps=1e-8)

  scheduler = None

  return model, optimizer, scheduler, p, m


def fix_model_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def NN_to_eval_mode(model, weight_path):
  ## Load weight
  model_state_dict = torch.load(weight_path)
  model.load_state_dict(fix_model_state_dict(model_state_dict))
  model.eval()
  return model


def plot_prediction(dl, conf_thres=0.001, nms_thres=.5, max_plot=5):
  batch = next(iter(dl))
  Xb, targets = batch
  '''
  check_with_plotting(Xb, targets)
  '''
  Xb = Xb[:100]
  targets = targets[:1000]
  Xb.cuda()
  # Xb = Xb[:4].cuda()
  # targets = targets[:20]
  outputs = model(Xb)
  outputs = non_max_suppression_1D(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
  plot_prediction_1D(Xb, targets, outputs, max_plot=max_plot)


def nms(global_outputs, nms_thres=0.):
    keep_first = True
    while global_outputs.size(0):
        # print(global_outputs.size(0))
        ii, is_overlap = 0, True
        while is_overlap:
            ii += 1
            if ii < len(global_outputs):
                is_overlap = global_outputs[ii, 0] <= global_outputs[0, 1]
            if ii == len(global_outputs):
                is_overlap = False
        idx_overlap_last =  ii - 1
        weights = global_outputs[0:idx_overlap_last+1, 2:3]
        max_obj_conf = weights.max()
        global_outputs[0, :2] = (weights * global_outputs[0:idx_overlap_last+1, :2]).sum(0) / weights.sum() # Merge
        global_outputs[0, 2] = max_obj_conf
        if keep_first:
            keep_ranges = global_outputs[0].unsqueeze(0)
            keep_first = False
        else:
            keep_ranges = torch.cat([keep_ranges, global_outputs[0].unsqueeze(0)], dim=0)
        global_outputs = global_outputs[idx_overlap_last+1:]
    return keep_ranges


def judge_overlap_existance(nms_outs):
    starts, ends = nms_outs[:,0], nms_outs[:,1]
    starts_shift_1, ends_shift_minus1 = nms_outs[1:,0], nms_outs[:-1,1]
    nms_finished = bool( (starts_shift_1 > ends_shift_minus1).all() )
    if not nms_finished:
        indi_false = np.where((starts_shift_1 - ends_shift_minus1) < 0)[0] + 1
        print(indi_false)
        print('Rest: {}'.format(len(indi_false)))
        return False
    else:
      return True

def print_SincNet_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'SincNet' in name:
                print(name, param.data)


## Parse File Paths
# Selected Ripples
# timestamps = {
# 'Test Mouse Number 01':'191127/201011/',
# 'Test Mouse Number 02':'191127/220722/',
# 'Test Mouse Number 03':'191127/233020/',
# 'Test Mouse Number 04':'191128/005420/',
# 'Test Mouse Number 05':'191128/021757/',
# }

# # Selected Binary
# timestamps = {
# 'Test Mouse Number 01':'191202/001835/',
# 'Test Mouse Number 02':'191202/015922/',
# 'Test Mouse Number 03':'191202/031853/',
# 'Test Mouse Number 04':'191202/044019/',
# 'Test Mouse Number 05':'191202/060250/',
# }

## With Label Confidence
# timestamps = {
# 'Test Mouse Number 01':'191206/004108/',
# 'Test Mouse Number 02':'191206/102729/',
# 'Test Mouse Number 03':'19120//',
# 'Test Mouse Number 04':'19120//',
# 'Test Mouse Number 05':'19120//',
# }

# 'Test Mouse Number 01':'191212/095446/', # 17 blocks, without ripple bandpass
# 'Test Mouse Number 01':'191212/153548/',
timestamps = {
'Test Mouse Number 01':'191212/232147/',
'Test Mouse Number 02':'19120//',
'Test Mouse Number 03':'19120//',
'Test Mouse Number 04':'19120//',
'Test Mouse Number 05':'19120//',
}


'''
d = mf.pkl_load('/mnt/md0/proj/results/191206/004108/epoch_1/batch_0/data.pkl')
# plt.plot(d['loss_tra'])
print(d['loss_tra'][-1])
'''

dirpath_root = '/mnt/md0/proj/results/{}'.format(timestamps['Test Mouse Number {}'.format(args.n_mouse_tes)])

## Load
# Load the last model for speculation
dirpaths = mf.natsorted_glob(dirpath_root + 'epoch_*/batch_*/')
lastdir = dirpaths[-1]
d, p, m, timer, weight_path, data_dir = load_params(lastdir)
plt.plot(d['loss_tra'][300:])

# Load most fitted model on train data
# i_loss_min = np.argmin(d['loss_tra'])
# dirpath_loss_min = dirpaths[i_loss_min]
# d, p, m, timer, weight_path, data_dir = load_params(dirpath_loss_min)
dirpath = dirpaths[1]
d, p, m, timer, weight_path, data_dir = load_params(dirpath)
assert p['tes_keyword'] == args.n_mouse_tes
print(d['loss_tra'])
print('Loss Min: {}'.format(np.min(d['loss_tra'])))

plt.plot(d['loss_tra'])

print('{} Epochs'.format(len(d['loss_tra'])/100))

# lfps, _ = load_lfps_rips_sec(p['fpaths_tes'])
lfps, ripples = load_lfps_rips_sec(p['fpaths_tes'][:2]) # fixme

## Parameters
SAMP_RATE = 1000
BS = 64 # 750
STEP = p['max_seq_len']


## Initiate NN
model, optimizer, scheduler, p, m = init_NN(p, m)
model = NN_to_eval_mode(model, weight_path)
print_SincNet_parameters(model)


## Main Loop
i_lfp=0
for i_lfp in range(len(lfps)):
    lfp, fpath_lfp = lfps[i_lfp], p['fpaths_tes'][i_lfp]
    lfp = lfp[:100*60*SAMP_RATE] # fixme, for developpment
    lfp_sliced = util.view_as_windows(lfp, window_shape=(p['max_seq_len'],), step=p['max_seq_len'])
    slice_starts_pts = np.array([i_slice*p['max_seq_len'] for i_slice in range(len(lfp_sliced))])
    n_batches = math.ceil(len(lfp_sliced) / BS)
    pred_hm_probs, pred_widths = [], []
    print('Collecting NN\'s outputs')
    with torch.no_grad():
        for i_batch in tqdm(range(n_batches)):
            Xb = torch.tensor(lfp_sliced[i_batch*BS:(i_batch+1)*BS]).cuda().unsqueeze(-1).float()
            _pred_hm_prob, _pred_width, pred_offset = model(Xb)
            pred_hm_probs += [_pred_hm_prob.detach().cpu().numpy()]
            pred_widths += [_pred_width.detach().cpu().numpy()]

    pred_hm_probs, pred_widths = np.hstack(np.vstack(pred_hm_probs)), np.hstack(np.vstack(pred_widths))
    plt.hist(pred_hm_probs, bins=1000)


    for _ in range(5):
        dt = STEP
        i = np.random.randint(len(lfp))
        plt.plot(lfp[i:i+dt])
        plt.plot(pred_hm_probs[i:i+dt])

        plt.pause(5)
        plt.close()

    plt.hist(pred_hm_probs, bins=1000)


#     '''
#     # mf.save_pkl(n_pattern_tot_outputs, '/mnt/md0/proj/_n_pattern_tot_outputs_selected_binary.pkl')
#     n_pattern_tot_outputs = mf.load_pkl('/mnt/md0/proj/_n_pattern_tot_outputs_selected_binary.pkl')
#      '''
#     ## Execute NMS on outputs of float64
#     print('Executing global NMS on {}-pattern NN\'s outputs'.format(N_PATTERN))
#     while not judge_overlap_existance(n_pattern_tot_outputs):
#         n_pattern_tot_outputs = nms(n_pattern_tot_outputs, NMS_THRES)
#     '''
#     obj_conf_thres = 0.
#     obj_conf = n_pattern_tot_outputs[:, 2]
#     cls_conf = n_pattern_tot_outputs[:, 3]
#     score = obj_conf * cls_conf
#     cls = n_pattern_tot_outputs[:, 4]
#     plt.hist(obj_conf[obj_conf > obj_conf_thres])
#     plt.hist(cls_conf[obj_conf > obj_conf_thres])
#     plt.hist(cls[obj_conf > obj_conf_thres])
#     plt.hist(score[obj_conf > obj_conf_thres])
#     # mf.save_pkl(n_pattern_tot_outputs, '/mnt/md0/proj/_n_pattern_tot_outputs_with_label_conf_nms.pkl')
#     n_pattern_tot_outputs = mf.load_pkl('/mnt/md0/proj/_n_pattern_tot_outputs_with_label_conf_nms.pkl')
#      '''

#     ## Save
#     ripples = pd.DataFrame({'start_sec':n_pattern_tot_outputs[:,0]/SAMP_RATE,
#                             'end_sec':n_pattern_tot_outputs[:,1]/SAMP_RATE,
#                             'obj_conf':n_pattern_tot_outputs[:,2],
#                             'cls_conf':n_pattern_tot_outputs[:,3],
#                             'cls_label':n_pattern_tot_outputs[:,4],
#                             'ripple_number':np.arange(len(n_pattern_tot_outputs))+1,
#                             })
#     ripples.set_index('ripple_number', inplace=True)
#     ripples['duration_ms'] = (ripples['end_sec'] - ripples['start_sec']) * 1000
#     ripples['score'] = ripples['obj_conf'] * ripples['cls_conf']


#     spath_ripples = fpath_lfp.replace('.npy', '_ripple_pred_with_label_conf_by_yolo.pkl')
#     # mf.pkl_save(ripples, spath_ripples)

# '''

# '''
