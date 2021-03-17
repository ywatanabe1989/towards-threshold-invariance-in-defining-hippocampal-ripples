## my own packages
import sys
sys.path.append('./')
import myutils.myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_centernet import DataloaderFulfiller
sys.path.append('./07_Learning/')
# from focal_loss import FocalLoss
from losses import FocalLoss
# from multi_task_loss import MultiTaskLoss
from optimizers import Ranger
from schedulers import cyclical_lr

from apex import amp
sys.path.append('./11_Models/')
sys.path.append('./11_Models/feature_extractors')
# from utils.utils import non_max_suppression_1D as nms
# from utils.utils import check_samples_1D, plot_prediction_1D


### others
import argparse
from collections import defaultdict
import datetime
import gc
import math
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.cluster import MiniBatchKMeans
import torch
import time
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
from pprint import pprint
import re
import terminalplot as tplt

import socket
hostname = socket.gethostname()
if hostname == 'localhost.localdomain':
  from delogger import Delogger
  Delogger.is_debug_stream = True
  debuglog = Delogger.line_profiler


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nmt", "--n_mouse_tes", default="01", choices=["01", "02", "03", "04", "05"], help = " ")
ap.add_argument("-o", "--use_opti", default='ranger', choices=['ranger', 'sdg'], help = " ")
# 12: Fast, 176: w/o 05/day5, 184: Full
ap.add_argument("-nl", "--n_load_all", default=184, choices=[12, 24, 184], type=int, help = " ")
ap.add_argument("-nul", "--n_use_lfp", default=10, type=int, help = " ")
ap.add_argument("-ca", "--calc_anchor", action='store_true', help = " ")
ap.add_argument("-sch", "--use_scheduler", default="none", nargs='+', choices=["none", "step", "clc"], help = " ")
ap.add_argument("-ilr", "--initial_learning_rate", default=1e-3, type=float, help=" ")
ap.add_argument("-bf", "--batch_factor", default=2, type=int, help=" ")
ap.add_argument("--model_def", default='./11_Models/yolo/config/yolov3.cfg', help=" ")
ap.add_argument("-sr", "--samp_rate", default=1000, type=int, help=" ")
# ap.add_argument("-msl", "--max_seq_len", default=416, choices=[408, 416], type=int, help=" ") # input_len, 408: wo bp
ap.add_argument("-wobp", "--without_ripple_bandpassing", action='store_true',  help=" ")
args = ap.parse_args()

# if args.without_ripple_bandpassing:
#     from _CenterNet_wo_padding import CenterNet1D_wo_bp as CenterNet1D
#     args.max_seq_len = 408
#     args.batch_factor = 3
# else:
#     from _CenterNet_wo_padding import CenterNet1D as CenterNet1D
#     args.max_seq_len = 416
#     args.batch_factor = 8

# args.max_seq_len = 992
# from CenterNet_after_SincNet import CenterNet1D_after_SincNet as CenterNet1D

args.max_seq_len = 1036
from CenterNet_191214 import CenterNet1D_after_SincNet as CenterNet1D, RESOLUTION_R

args.calc_anchor = True
args.use_fp16 = True


def init_params():
    ## Preparation
    TS = datetime.datetime.fromtimestamp(time.time()).strftime('%y%m%d/%H%M%S')
    print('Time Stamp {}'.format(TS))
    d, p, m = defaultdict(list), defaultdict(list), defaultdict(list)
    timer = mf.time_tracker()

    p['time_stamp'] = TS

    ## CUDA
    m['use_cuda'] = torch.cuda.is_available()
    m['device'] = torch.device("cuda:0" if  m['use_cuda'] else "cpu")
    m['n_gpus'] = int(torch.cuda.device_count())
    print('n_GPUs: {}'.format(m['n_gpus']))

    ## Parse File Pasths
    LPATHS_NPY_LIST = '../data/1kHz_npy_list.pkl'
    p['n_load_all'] = args.n_load_all # 184   # 12 -> fast, 176 -> w/o 05/day5, 184 -> full
    FPATHS = mf.pkl_load(LPATHS_NPY_LIST)[:p['n_load_all']]
    p['tes_keyword'] = args.n_mouse_tes # '02'
    p['fpaths_tra'], p['fpaths_tes'] = mf.split_fpaths(FPATHS, tes_keyword=p['tes_keyword'])
    print('n_load_all: {}'.format(p['n_load_all']))
    print('Test Keyword: {}'.format(p['tes_keyword']))

    ## Parameters
    p['samp_rate'] = args.samp_rate
    p['max_seq_len'] = args.max_seq_len
    p['use_perturb_tra'] = True
    p['batch_factor'] = args.batch_factor
    p['bs_tra'] = 64 * m['n_gpus'] * p['batch_factor']
    p['bs_tes'] = 64 * m['n_gpus'] * p['batch_factor']
    print('Batchsize: Train {}, Test {}'.format(p['bs_tra'], p['bs_tes']))

    # Intervals
    p['max_epochs'] = 1000
    p['test_epoch_interval'] = 1
    p['n_print_and_save_per_epoch'] = 100

    # NN
    # General
    m['init_lr'] = args.initial_learning_rate * p['batch_factor']
    # Save switches
    m['use_fp16'] = args.use_fp16 # fixme
    m['use_scheduler'] = args.use_scheduler
    # Architecture
    m['n_features'] = 1
    m['use_opti'] = args.use_opti
    m['use_exclusive'] = True
    m['use_label_conf'] = True

    return d, p, m, timer


def init_train_dataloader(d, p, m):
    print('Initializing Train Dataloader')
    keys_to_pack = ['Xb', 'Tb_c_pad', 'Tb_w_pad']

    kwargs_dl = {'samp_rate':p['samp_rate'],
                 'use_fp16':m['use_fp16'],
                 'use_shuffle':True,
                 'max_seq_len_pts':p['max_seq_len'],
                 'step':None,
                 'use_perturb':True,
                 'define_ripples':True,
                 'keys_to_pack':keys_to_pack,
                 'bs':p['bs_tra'],
                 'nw':20,
                 'pm':False, # RuntimeError: Caught RuntimeError in pin memory thread for device 0.
                 'drop_last':True,
                 'n_mouse_tes':p['tes_keyword'],
                 'smoothing_sigma':0.004,
                 'R':RESOLUTION_R,
                 }

    dl_fulf_tra = DataloaderFulfiller(p['fpaths_tra'], n_use_lfp=args.n_use_lfp, **kwargs_dl)
    d['n_samples_tra'] = dl_fulf_tra.get_n_samples()
    d['n_batches_per_epoch'] = math.floor(d['n_samples_tra'] / p['bs_tra'])

    p['print_and_save_batch_interval'] = max(1, math.floor(d['n_batches_per_epoch']/p['n_print_and_save_per_epoch']))

    return dl_fulf_tra, d, p, m


def init_NN(p, m):
    print('Initializing NN')

    model = CenterNet1D().to(m['device'])
    '''
    CenterNet cannot handle FP16 precision.
    '''

    optimizer = Ranger(model.parameters(), lr=m['init_lr'], eps=1e-8)

    if m['n_gpus'] > 1:
      model = torch.nn.DataParallel(model).to(m['device'])

    scheduler = None

    if m['use_scheduler'] == 'step':
      import pdb; pdb.set_trace()
      step_all = d['n_batches_per_epoch'] * p['max_epochs']
      last_relat_lr = 0.1
      gamma = np.exp(np.log(last_relat_lr) / step_all)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) # 10.0

    if m['use_scheduler'] == 'clc':
      step_epochs = 2
      step_size = int(d['n_samples_tra'] / p['bs_tra']) * step_epochs # 100
      clc = cyclical_lr(step_size, min_lr=1.0, max_lr=10.0)
      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clc])


    return model, optimizer, scheduler, p, m


def _save_tra(i_batch, optimizer, d, p, m):

  d['lr/batch_factor'].append(mf.get_lr(optimizer) / p['batch_factor'])

  save_dir = '/mnt/md0/proj/results/{}/epoch_{}/batch_{}/'.format(p['time_stamp'], d['epoch'][-1], i_batch)
  os.makedirs(save_dir, exist_ok=True)
  save_names = ['weight.pth', 'opti_params.pth', 'data.pkl', 'model.pkl', 'params.pkl'] # , 'functions.dill']
  save_items = [model.state_dict(), optimizer.state_dict(), d, m, p] # , f]
  for (save_name, save_item) in zip(save_names, save_items):
    _, _, ext = mf.split_fpath(save_name)
    spath = save_dir + save_name
    if ext == '.pth':
      torch.save(save_item, spath)
      print('Saved to: {}'.format(spath))
    if ext == '.pkl':
      mf.pkl_save(save_item, spath)
    if ext == '.dill':
      mf.dill_save(save_item, spath)

def print_SincNet_parameters(model):
    _softmax = torch.nn.Softmax(dim=1)
    _sigmoid = torch.nn.Sigmoid()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'SincNet' in name:
                print(name)
                print(list(abs(param.detach().cpu().numpy().squeeze().astype(np.int))))
                print()
            if 'bands_mix_weights' in name:
                print(name)
                print(list(_sigmoid(param).detach().cpu().numpy().squeeze()))
                print()

def term_plot(y_arr, rows=10, columns=200):
       import terminalplot as tplt
       x_arr = np.arange(len(y_arr))
       tplt.plot(list(x_arr), list(y_arr), rows=rows, columns=columns)

def train(model, dl_fulf_tra, d, p, m):
    global dl_tra
    model.train()
    dl_tra = dl_fulf_tra()
    kldiv_criterion = torch.nn.KLDivLoss()
    focal_criterion = FocalLoss(alpha=2, beta=4) # for hm # default: alpha=2, beta=4
    # The bigger beta is, the more weight are on background.
    l1_criterion = torch.nn.L1Loss(reduction='batchmean') # for width, offset
    for i_batch, batch in enumerate(dl_tra):
        '''
        i_batch, batch = next(enumerate(dl_tra))
        '''
        Xb, Tb_hm, Tb_w, Tb_hm_R, Tb_w_R = batch
        Xb, Tb_hm_R, Tb_w_R = Xb.to(m['device']), Tb_hm_R.to(m['device']), Tb_w_R.to(m['device'])
        Xb, Tb_hm_R, Tb_w_R = Xb.float(), Tb_hm_R.float(), Tb_w_R.float()

        optimizer.zero_grad()

        pred_hm_prob, pred_width, pred_offset = model(Xb.unsqueeze(-1))

        # loss_hm = kldiv_criterion(torch.log(pred_hm_prob + 1e-5).view(-1,1), Tb_hm_R.view(-1,1)).mean()
        loss_hm = focal_criterion(pred_hm_prob.view(-1,1), Tb_hm_R.view(-1,1)).mean()

        w_mask = (Tb_w_R != 0)
        loss_w = l1_criterion(Tb_w_R[w_mask].float(), pred_width[w_mask].float())

        lambda_w = 0.1 # fixme
        loss = loss_hm + lambda_w * loss_w

        loss.backward()

        optimizer.step()

        if i_batch % p['print_and_save_batch_interval'] == 0:
            print()
            # Loss
            print('--- Loss ---')
            _loss = float(loss.detach().cpu().numpy())
            print(np.mean(_loss))
            d['loss_tra'].append(_loss)
            term_plot(np.array(d['loss_tra']))
            print('------------')
            print()

            # Heatmaps
            i_print = w_mask.any(dim=-1).argmax()
            print('--- Heat Maps ---')
            print('(Ground Truth)')
            term_plot(Tb_hm_R[i_print].detach().cpu().numpy())
            print()
            print('(Prediction)')
            term_plot(pred_hm_prob[i_print].detach().cpu().numpy())
            print('Max in this Batch: {}'.format(pred_hm_prob.max().detach().cpu().numpy()))
            print('Min in this Batch: {}'.format(pred_hm_prob.min().detach().cpu().numpy()))
            print('------------')
            print()

            # Learning Rate
            print('Current Learning Rate / Batch Factor: {}'.format(mf.get_lr(optimizer) / p['batch_factor'])) # scheduler.get_lr
            print()

            # SincNet Parameters
            print_SincNet_parameters(model)
            print()

            # Save
            _save_tra(i_batch, optimizer, d, p, m)

        if (m['use_scheduler'] == 'clc') or (m['use_scheduler'] == 'step'):
            scheduler.step()


def main(model, dl_fulf_tra, d, p, m, timer):
  for epoch in range(1, p['max_epochs']+1):
    d['epoch'].append(epoch)
    timer('Epoch {} Training starts'.format(d['epoch'][-1]))
    train(model, dl_fulf_tra, d, p, m)
    timer('Epoch {} Training ends'.format(d['epoch'][-1]))
    d['time_by_epoch'].append(timer.from_prev_hhmmss)


# def _check_samples(Xb, targets, label_conf=None, max_plot=1):
#     plot = 0
#     for _ in range(1000):
#       ix = np.random.randint(len(Xb))
#       x = Xb[ix].squeeze()
#       indi_targets = (targets[:,0] == ix)
#       _targets = targets[indi_targets]
#       classes = _targets[:,1]

#       # Title
#       if label_conf is not None:
#           _label_conf = label_conf[indi_targets]
#           title = 'Classes: {} \n Confidence: {:.2f}'.format(classes.long().item(), _label_conf.squeeze().item())
#       else:
#           title = 'Classes: {}'.format(classes.long().item())

#       fig, ax = plt.subplots()
#       ax.plot(np.arange(len(x)), x)

#       for i_target in range(len(_targets)):
#         obj = _targets[i_target]
#         # cls, X, W = obj[1], obj[2], obj[3]
#         cls, X, W = obj[:3].T()
#         X, W = X*len(x), W*len(x)
#         left, right = int(X - W/2), int(X + W/2)
#         # left = int(X - W/2)
#         # right = int(X + W/2)
#         ax.axvspan(left, right, alpha=0.3, color='red', zorder=1000)

#       plt.title(title)

#       plot += 1
#       if plot == max_plot:
#         break

# def _check_outputs(outputs, Xb, targets, label_conf=None, max_plot=1):
#     plot = 0
#     for _ in range(1000):
#       ix = np.random.randint(len(Xb))
#       x = Xb[ix].squeeze()
#       indi_targets = (targets[:,0] == ix)
#       _targets = targets[indi_targets]
#       classes = _targets[:,1]

#       # Title
#       if label_conf is not None:
#           _label_conf = label_conf[indi_targets]
#           title = 'Classes: {} \n Label Conf.: {}'.format(classes.long(), _label_conf.squeeze())
#       else:
#           title = 'Classes: {}'.format(classes.long().item())
#       fig, ax = plt.subplots()

#       # LFP
#       ax.plot(np.arange(len(x)), x)

#       # Targets
#       for i_target in range(len(_targets)):
#         obj = _targets[i_target]
#         cls, X, W = obj[1], obj[2], obj[3]
#         X, W = X*len(x), W*len(x)
#         left, right = int(X - W/2), int(X + W/2)
#         ax.axvspan(left, right, alpha=0.3, color='red', zorder=1000)
#         # text = 'R: {:.2f}'.format(_ripple.ripple_prob_by_ResNet)
#         # ax.text((_ripple.start_sec+_ripple.end_sec)/2-0.01, ymax-100, text)

#       # Outputs
#       _outputs = outputs[ix]
#       if _outputs is not None:
#           for i_output in range(len(_outputs)):
#             obj = _outputs[i_output]
#             left, right, obj_conf, cls_conf, cls = obj.t()
#             ax.axvspan(left, right, alpha=0.3, color='black', zorder=1000)
#             text = 'cls: {}\ncls_conf: {:.2f}\nobj_conf: {:.2f}'.format(cls, cls_conf, obj_conf)
#             h = 100
#             h = h if cls == 1 else -h
#             ax.text((left+right)/2, h, text)

#       plt.title(title)

#       plot += 1
#       if plot == max_plot:
#         break


# def plot_prediction(dl_tra):
#   from pprint import pprint
#   batch = next(iter(dl_tra))
#   Xb, targets_with_label_conf = batch
#   Xb = Xb[:8].cuda()
#   targets_with_label_conf = targets_with_label_conf[:10]
#   '''
#   _check_samples(Xb.cpu(),
#                  targets_with_label_conf[:, :4],
#                  label_conf=targets_with_label_conf[:,4])
#   '''
#   outputs = model(Xb)
#   outputs = nms(outputs, conf_thres=0.1, nms_thres=0.)

#   _check_outputs(outputs,
#                  Xb.cpu(),
#                  targets_with_label_conf[:, :4],
#                  label_conf=targets_with_label_conf[:,4],
#                  max_plot=3
#   )


#   print(targets)
#   pprint(outputs)
#   plot_prediction_1D(Xb, targets, outputs, max_plot=3)


if __name__ == "__main__":
  d, p, m, timer = init_params()
  dl_fulf_tra, d, p, m = init_train_dataloader(d, p, m)
  m['anchors'] = None
  model, optimizer, scheduler, p, m = init_NN(p, m)
  main(model, dl_fulf_tra, d, p, m, timer)

  '''
  plot_prediction(dl_tra)

  dl_tes = dl_fulf_tes()
  plot_prediction(dl_tes)
  '''
