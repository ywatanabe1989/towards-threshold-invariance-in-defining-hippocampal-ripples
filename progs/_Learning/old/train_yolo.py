## my own packages
import sys
sys.path.append('./')
# sys.path.append('./utils')
import myutils.myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_yolo_191028 import dataloader_fulfiller
sys.path.append('./07_Learning/')
from optimizers import Ranger
from schedulers import cyclical_lr
from apex import amp
sys.path.append('./11_Models/')
sys.path.append('./11_Models/yolo')
sys.path.append('./11_Models/yolo/utils')
from yolo.models import Darknet
from yolo.data_parallel import DataParallel
from utils.utils import non_max_suppression_1D as nms

### others
import argparse
from collections import defaultdict
import datetime
import gc
import math
import numpy as np
import os
from sklearn.utils import shuffle
import torch
import time
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
from pprint import pprint
import re

import socket
hostname = socket.gethostname()
if hostname == 'localhost.localdomain':
  from delogger import Delogger
  Delogger.is_debug_stream = True
  debuglog = Delogger.line_profiler


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nmt", "--n_mouse_tes", default="01", choices=["01", "02", "03", "04", "05"], help = " ")
ap.add_argument("-o", "--use_opti", default='ranger', choices=['ranger', 'sdg'], help = " ")
ap.add_argument("-nl", "--n_load_all", default=12, choices=[12, 24, 186], type=int, help = " ")
ap.add_argument("-bi", "--batch_interval", default=100, type=int, help=" ")
ap.add_argument("-sch", "--use_scheduler", default="none", nargs='+', choices=["none", "step", "clc"], help = " ")
ap.add_argument("-ilr", "--initial_learning_rate", default=1e-3, type=float, help=" ")
ap.add_argument("-bf", "--batch_factor", default=1, type=int, help=" ")
ap.add_argument("--model_def", default='./11_Models/yolo/config/yolov3.cfg', help=" ")
# ap.add_argument("--input_len", default=416, type=int, choices=[52, 104, 208, 416], help=" ")
ap.add_argument("-sr", "--samp_rate", default=1000, type=int, help=" ")
ap.add_argument("-msl", "--max_seq_len", default=416, choices=[52, 104, 384, 416], type=int, help=" ")
args = ap.parse_args()


'''
from fp16util import network_to_half
for i in range(2, 12):
  input_len = 32*i
  model = Darknet(args.model_def, dim=1, n_classes=1, input_len=input_len).cuda()
  model = network_to_half(model)
  x = torch.rand(1,1,input_len).half()
  # %timeit x.cuda()
  %timeit model(x.cuda())
  print(input_len)
'''


args.use_fp16 = True


# def calc_normalization_params(N):
#   print('Calculating parameters for normalization...')

#   updater = mf.update_mean_var_std_arr()
#   for _ in range(N):

#     dl_tra = dl_fulf_tra.fulfill()
#     for i_batch, batch in tqdm(enumerate(dl_tra)):
#       _, _, Tb_lat = batch
#       Tb_lat = Tb_lat.cpu().numpy()
#       Tb_lat.astype(np.float64)
#       log_lat_mean, _, log_lat_std = updater.update(np.log(Tb_lat + 1e-5))
#   p['log(lat+1e-5)_mean'] = log_lat_mean
#   p['log(lat+1e-5)_std'] = log_lat_std[0]

def define_parameters():
  # global ts, d, p, m, timer
  ## Preparation
  ts = datetime.datetime.fromtimestamp(time.time()).strftime('%y%m%d/%H%M%S')
  print('Time Stamp {}'.format(ts))
  d = defaultdict(list)
  p = defaultdict(list)
  m = defaultdict(list)
  timer = mf.time_tracker()

  ## CUDA for PyTorch
  m['use_cuda'] = torch.cuda.is_available()
  m['device'] = torch.device("cuda:0" if  m['use_cuda'] else "cpu")
  m['n_gpus'] = int(torch.cuda.device_count())
  print('n_GPUs: {}'.format(m['n_gpus']))

  ## Load Paths
  loadpath_npy_list = '../data/1kHz_npy_list.pkl'
  p['n_load_all'] = args.n_load_all # 186   # 12 -> fast, 176 -> w/o 05/day5, 186 -> full
  print('n_load_all: {}'.format(p['n_load_all']))
  fpaths = mf.pkl_load(loadpath_npy_list)[:p['n_load_all']]
  p['tes_keyword'] = args.n_mouse_tes # '02'
  print('Test Keyword: {}'.format(p['tes_keyword']))
  p['fpaths_tra'], p['fpaths_tes'] = mf.split_fpaths(fpaths, tes_keyword=p['tes_keyword'])

  ## Parameters
  p['samp_rate'] = args.samp_rate
  p['max_seq_len'] = args.max_seq_len
  p['use_perturb_tra'] = True
  p['batch_factor'] = args.batch_factor
  p['bs_tra'] = 64 * m['n_gpus'] * p['batch_factor']
  print('Batchsize: Train {}, Test {}'.format(p['bs_tra'], p['bs_tes']))

  # Intervals
  p['max_epochs'] = 3
  p['test_epoch_interval'] = 1
  p['save_batch_interval'] = args.batch_interval # These intervals should be calculated from the sample size and batch size.
  p['print_log_batch_interval'] = args.batch_interval
  p['lr_scheduler_batch_interval'] = args.batch_interval

  # NN
  # General
  m['init_lr'] = args.initial_learning_rate * p['batch_factor']
  # Save switches
  m['use_fp16'] = args.use_fp16 # fp16 is not used in Darknet now.
  m['use_scheduler'] = args.use_scheduler
  # Architecture
  m['n_features'] = 1
  m['use_opti'] = args.use_opti
  return ts, d, p, m, timer

def init_dataloaders(p, m):
  # global dl_fulf_tra
  keys_to_pack_ripple_detect = ['Xb', 'Tb_CenterX_W', 'Tb_levels']
  # Targets should be converted to relative coordinates before feeding to NN (, or as the output of dataloader)
  kwargs_dl_tra = {'samp_rate':p['samp_rate'],
                   'use_fp16':m['use_fp16'],
                   'use_shuffle':True,
                   'max_seq_len_pts':p['max_seq_len'],
                   'step':None,
                   'use_perturb':True,
                   'define_ripples':True,
                   'keys_to_pack':keys_to_pack_ripple_detect,
                   'bs':p['bs_tra'],
                   'nw':20,
                   'pm':False, # RuntimeError: Caught RuntimeError in pin memory thread for device 0.
                   'drop_last':True,
                   }

  dl_fulf_tra = dataloader_fulfiller(p['fpaths_tra'], **kwargs_dl_tra)
  # d['n_samples_tra'] = dl_fulf_tra.get_n_samples()
  # d['n_batches_per_epoch'] = math.ceil(d['n_samples_tra'] / p['bs_tra'])
  # global dl_fulf_tra, dl_fulf_tes
  return dl_fulf_tra


def init_ripple_detector_NN(p, m):
  # global model, optimizer, scheduler

  model = Darknet(args.model_def, dim=1, n_classes=1, input_len=args.max_seq_len).to(m['device'])

  # fixme
  # if m['use_fp16']:
  #   adam_eps = 1e-4
  #   if args.use_opti == 'ranger':
  #     optimizer = Ranger(learnable_params, lr=m['init_lr'], eps=adam_eps)
  #   if args.use_opti == 'sgd':
  #     optimizer = torch.optim.SGD(learnable_params, lr=m['init_lr'], nesterov=False)
  #   model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=num_losses)
  # else:
  #   adam_eps = 1e-8
  #   if args.use_opti == 'ranger':
  #     optimizer = Ranger(learnable_params, lr=m['init_lr'], eps=adam_eps)
  #   if args.use_opti == 'sgd':
  #     optimizer = torch.optim.SGD(learnable_params, lr=m['init_lr'], nesterov=False)
  adam_eps = 1e-8
  optimizer = Ranger(model.parameters(), lr=m['init_lr'], eps=adam_eps)

  if m['n_gpus'] > 1:
    # model = torch.nn.DataParallel(model).to(m['device'])
    model = DataParallel(model).to(m['device'])

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
  return model, optimizer, scheduler



def _calc_ave(cum_dict):
  keys = cum_dict.keys()

  n_counter_keys = mf.search_str_list(keys, 'n_counter')
  n_counter = cum_dict[n_counter_keys[0]]

  cum_loss_keys = mf.search_str_list(keys, 'cum_loss_')

  for k in cum_loss_keys:
    cum_dict['ave_'+k.replace('cum_', '')] = 1.0 * cum_dict[k] / n_counter

  return cum_dict


def _print_log_tra(cum_dict):
  print('-----------------------------------------')
  print('\nTrain Epoch: {}'.format(d['epoch'][-1]))
  pprint(cum_dict)
  print('Current Learning Rate / Batch Factor: {}'.format(mf.get_lr(optimizer) / p['batch_factor'])) # scheduler.get_lr
  print('-----------------------------------------')


def _save_tra(i_batch, cum_dict):
  keys = cum_dict.keys()
  loss_keys = mf.search_str_list(keys, 'loss')
  for k in loss_keys:
    d[k].append(cum_dict[k])

  d['conf_mat_tra'].append(cum_dict['conf_mat_tra'])
  d['cls_report_tra'].append(cum_dict['cls_report_tra'])
  d['lr/batch_factor'].append(mf.get_lr(optimizer) / p['batch_factor'])

  save_dir = '../results/{}/epoch_{}/batch_{}/'.format(ts, d['epoch'][-1], i_batch)
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


# @Delogger.line_memory_profiler
def train():
  d['epoch'] = [1] if not d.get('epoch') else d.get('epoch')
  timer('Epoch {} Train starts'.format(d['epoch'][-1]))
  model.train()
  keys_cum = ['cum_loss_tra', 'n_counter_tra']
  cum_dict = mf.init_dict(keys_cum)
  dl_tra = dl_fulf_tra.fulfill()
  cls_rec = mf.class_recorder()

  # for i_batch in range(d['n_batches_per_epoch']):
  #   batch = next(iter(dl_tra))
  for i_batch, batch in enumerate(dl_tra):
    '''
    dl_tra = dl_fulf_tra.fulfill()
    batch = next(iter(dl_tra))
    '''
    Xb, targets = batch
    Xb, targets = Xb.to(m['device']), targets.to(m['device'])

    '''
    check_with_plotting(Xb, targets, max_plot=10)
    '''
    # import pdb; pdb.set_trace()
    targets[:,1] = 0 # fixme
    optimizer.zero_grad()

    # print(i_batch, Xb.shape, targets.shape)
    loss, output, metrics = model(Xb, targets=targets) # [B, C, W]

    # losses
    # if m['apply_n_labels_balancing']:
    if False: # fixme
      loss = balance_loss(loss_isRipple, Tb_isRipple) # Scaling wrt sample sizes

    loss.backward()
    optimizer.step()

    cum_dict['cum_loss_tra'] += loss.detach().cpu().numpy().astype(np.float)
    cum_dict['n_counter_tra'] += len(Xb)

    print(loss)
  #   if i_batch % p['print_log_batch_interval'] == 0:
  #     cum_dict = _calc_ave(cum_dict)
  #     cum_dict['conf_mat_tra'], cum_dict['cls_report_tra'] = cls_rec.report(labels_cm=['noRipple', 'Ripple'], labels_cr=[0, 1])
  #     _print_log_tra(cum_dict)

  #   if i_batch % p['save_batch_interval'] == 0:
  #     _save_tra(i_batch, cum_dict)
  #     cum_dict = mf.init_dict(keys_cum)

  #   if (m['use_scheduler'] == 'clc') or (m['use_scheduler'] == 'step'):
  #     scheduler.step()

  # if d['epoch'][-1] == 1:
  #   d['n_tra'] = cum_dict['n_counter_tra']

  # del dl_tra, batch, Xb, Tb_isRipple, Xb_Tbs_dict; gc.collect()

  # timer('Epoch {} Train ends'.format(d['epoch'][-1]))

def check_with_plotting(Xb, targets, max_plot=1):
    Xb = Xb.cpu().numpy()
    targets = targets.clone()

    plot = 0
    for _ in range(1000):
      ix = np.random.randint(len(Xb))
      x = Xb[ix].squeeze()
      indi_targets = (targets[:,0] == ix)
      _targets = targets[indi_targets].clone()
      classes = _targets[:,1]

      fig, ax = plt.subplots()
      ax.plot(np.arange(len(x)), x)

      for i_target in range(len(_targets)):
        obj = _targets[i_target]
        cls, X, W = obj[1], obj[2], obj[3]
        X, W = X*len(x), W*len(x)
        left = int(X - W/2)
        right = int(X + W/2)
        ax.axvspan(left, right, alpha=0.3, color='red', zorder=1000)

      # ax.text(0,0, txt, transform=ax.transAxes)
      plt.title('Classes {}'.format(classes))

      plot += 1
      if plot == max_plot:
        break


def plot_prediction(Xb, targets, outputs, max_plot=1):
    targets = targets.clone()
    outputs = outsputs.clone()

    targets[:,2:] *= Xb.shape[-1]
    # max_plot = 3
    plot = 0
    for _ in range(1000):
      ix = np.random.randint(len(Xb))
      x = Xb[ix].squeeze()
      indi_targets = (targets[:,0] == ix)
      _targets = targets[indi_targets]
      _outputs = outputs[ix]

      gt_cls = _targets[:,1]

      fig, ax = plt.subplots()
      ax.plot(np.arange(len(x)), x.cpu().numpy())

      for i_target in range(len(_targets)):
        obj = _targets[i_target]
        cls, X, W = obj[1], obj[2], obj[3]
        left = int(X - W/2)
        right = int(X + W/2)
        ax.axvspan(left, right, alpha=0.3, color='red', zorder=1000, label='GT')

      for i_output in range(len(_outputs)):
        obj = _outputs[i_output]
        cls, X, W = obj[-1], obj[0], obj[1]
        left = int(X - W/2)
        right = int(X + W/2)
        ax.axvspan(left, right, alpha=0.3, color='blue', zorder=1000, label='pred')

      plt.title('Gt_cls {}'.format(gt_cls))

      plot += 1
      if plot == max_plot:
        break


def main():
  # for epoch in range(1, p['max_epochs']+1):
  for epoch in range(1, 1000+1):
    d['epoch'].append(epoch)
    train()
    d['time_by_epoch'].append(timer.from_prev_hhmmss)




if __name__ == "__main__":
  ts, d, p, m, timer = define_parameters()
  dl_fulf_tra = init_dataloaders(p, m)
  anchors = calc_anchors(dl_fulf_tra)
  model, optimizer, scheduler = init_ripple_detector_NN(anchors)

  #   # calc_normalization_params(3)
  # # datadir = '../results/191014/075644/epoch_10/batch_29600/'
  # # if datadir:
  # #   load_parameters(datadir)
  main()


  '''
  dl_tra = dl_fulf_tra.fulfill()
  batch = next(iter(dl_tra))
  Xb, targets = batch
  Xb = Xb[:4].cuda()
  targets = targets[:10]

  outputs = model(Xb)
  outputs = nms(outputs, conf_thres=.8, nms_thres=.5)

  plot_prediction(Xb, targets, outputs, max_plot=3)
  '''
