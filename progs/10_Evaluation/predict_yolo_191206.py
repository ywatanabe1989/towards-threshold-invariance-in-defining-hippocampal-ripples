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
def load_params(dirpath_root, plot=False):
  timer = mf.time_tracker()
  last_dir = mf.natsorted_glob(dirpath_root + 'epoch_*/batch_*/')[-1]
  print('Data directory: {}'.format(last_dir))
  d = mf.pkl_load(last_dir + 'data.pkl')
  p = mf.pkl_load(last_dir + 'params.pkl')
  m = mf.pkl_load(last_dir + 'model.pkl')
  p['bs_tes'] = 1024 * 10
  weight_path = last_dir + 'weight.pth'
  return d, p, m, timer, weight_path, last_dir


def init_NN(p, m):
  print('Initializing NN')
  model = Darknet(args.model_def,
                  dim=1,
                  n_classes=2,
                  input_len=args.max_seq_len,
                  anchors=m['anchors'],
                  ).to(m['device'])

  adam_eps = 1e-8
  optimizer = Ranger(model.parameters(), lr=m['init_lr'], eps=adam_eps)

  scheduler = None

  if m['use_scheduler'] == 'step':
    step_all = d['n_batches_per_epoch'] * p['max_epochs']
    last_relat_lr = 0.1
    gamma = np.exp(np.log(last_relat_lr) / step_all)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) # 10.0

  if m['use_scheduler'] == 'clc':
    step_epochs = 2
    step_size = (n_tra / p['bs_tra']) * step_epochs # 100
    clc = cyclical_lr(step_size, min_lr=1.0, max_lr=10.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clc])

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
timestamps = {
'Test Mouse Number 01':'191206/004108/',
'Test Mouse Number 02':'191206/102729/',
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
d, p, m, timer, weight_path, data_dir = load_params(dirpath_root)
'''
d, p, m, timer, weight_path, data_dir = load_params(dirpath_root)
plt.plot(d['loss_tra'][1:])
print(np.min(d['loss_tra'][-1]))

'''
lfps, _ = load_lfps_rips_sec(p['fpaths_tes'])

## Parameters
SAMP_RATE = 1000
CONF_THRES, NMS_THRES = 1e-3, 0. # 1e-10
BS = 64 # 750
SLIDE_PERCENT = 25
N_PATTERN = int(100/SLIDE_PERCENT)
STEP = int(p['max_seq_len'] * SLIDE_PERCENT / 100)


## Initiate NN
model, optimizer, scheduler, p, m = init_NN(p, m)
model = NN_to_eval_mode(model, weight_path)


## Main Loop
for i_lfp in range(len(lfps)):
    lfp, fpath_lfp = lfps[i_lfp], p['fpaths_tes'][i_lfp]
    lfp = lfp[:100*60*SAMP_RATE] # fixme, for developpment
    n_pattern_tot_outputs = []
    print('Getting NN\'s outputs')
    for i_pattern in range(N_PATTERN): # Get NN's outputs
        start_pts = STEP*i_pattern
        lfp_sliced = util.view_as_windows(lfp[start_pts:], window_shape=(p['max_seq_len'],), step=p['max_seq_len'])
        slice_starts = np.array([start_pts + i_slice*p['max_seq_len'] for i_slice in range(len(lfp_sliced))])
        n_batches = math.ceil(len(lfp_sliced) / BS)
        nn_outputs = []
        for i_batch in tqdm(range(n_batches)):
            Xb = torch.tensor(lfp_sliced[i_batch*BS:(i_batch+1)*BS]).cuda()
            _nn_outputs = model(Xb.cuda().unsqueeze(1).float())
            _nn_outputs = non_max_suppression_1D(_nn_outputs, conf_thres=CONF_THRES, nms_thres=NMS_THRES)
            nn_outputs += _nn_outputs

        for i in range(len(nn_outputs)): # Concat to one big array
            if type(nn_outputs[i]) == type(None):
                nn_outputs[i] = torch.tensor(np.ones(5)*np.nan, dtype=torch.float)
            else:
                nn_outputs[i] = nn_outputs[i].view(-1)

        nn_outputs = torch.nn.utils.rnn.pad_sequence(nn_outputs, batch_first=True, padding_value=np.nan)

        slice_starts = torch.FloatTensor(slice_starts) # Adjust starts
        for i in range(nn_outputs.shape[-1]):
            if (i % 5 == 0) or (i % 5 == 1):
                nn_outputs[:, i] += slice_starts

        nn_outputs = nn_outputs.view(-1) # Filter out NaN and Reshape
        nn_outputs = nn_outputs[~torch.isnan(nn_outputs)].view(-1, 5)

        n_pattern_tot_outputs.append(nn_outputs)

    n_pattern_tot_outputs = torch.cat(n_pattern_tot_outputs)
    indi = n_pattern_tot_outputs[:,0].argsort()
    n_pattern_tot_outputs = n_pattern_tot_outputs[indi].double() # double precision is needed for the next NMS calc.

    '''
    # mf.save_pkl(n_pattern_tot_outputs, '/mnt/md0/proj/_n_pattern_tot_outputs_selected_binary.pkl')
    n_pattern_tot_outputs = mf.load_pkl('/mnt/md0/proj/_n_pattern_tot_outputs_selected_binary.pkl')
     '''
    ## Execute NMS on outputs of float64
    print('Executing global NMS on {}-pattern NN\'s outputs'.format(N_PATTERN))
    while not judge_overlap_existance(n_pattern_tot_outputs):
        n_pattern_tot_outputs = nms(n_pattern_tot_outputs, NMS_THRES)
    '''
    obj_conf_thres = 0.
    obj_conf = n_pattern_tot_outputs[:, 2]
    cls_conf = n_pattern_tot_outputs[:, 3]
    score = obj_conf * cls_conf
    cls = n_pattern_tot_outputs[:, 4]
    plt.hist(obj_conf[obj_conf > obj_conf_thres])
    plt.hist(cls_conf[obj_conf > obj_conf_thres])
    plt.hist(cls[obj_conf > obj_conf_thres])
    plt.hist(score[obj_conf > obj_conf_thres])
    # mf.save_pkl(n_pattern_tot_outputs, '/mnt/md0/proj/_n_pattern_tot_outputs_with_label_conf_nms.pkl')
    n_pattern_tot_outputs = mf.load_pkl('/mnt/md0/proj/_n_pattern_tot_outputs_with_label_conf_nms.pkl')
     '''

    ## Save
    ripples = pd.DataFrame({'start_sec':n_pattern_tot_outputs[:,0]/SAMP_RATE,
                            'end_sec':n_pattern_tot_outputs[:,1]/SAMP_RATE,
                            'obj_conf':n_pattern_tot_outputs[:,2],
                            'cls_conf':n_pattern_tot_outputs[:,3],
                            'cls_label':n_pattern_tot_outputs[:,4],
                            'ripple_number':np.arange(len(n_pattern_tot_outputs))+1,
                            })
    ripples.set_index('ripple_number', inplace=True)
    ripples['duration_ms'] = (ripples['end_sec'] - ripples['start_sec']) * 1000
    ripples['score'] = ripples['obj_conf'] * ripples['cls_conf']


    spath_ripples = fpath_lfp.replace('.npy', '_ripple_pred_with_label_conf_by_yolo.pkl')
    # mf.pkl_save(ripples, spath_ripples)

'''

'''
