### my own packages
import sys
sys.path.append('./')
sys.path.append('./utils')
import myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_191009 import dataloader_fulfiller_detect
sys.path.append('./07_Learning/')
from balance_loss import balance_loss
from ripple_detect_loss import get_lambda
# from pdf_loss import pdf_loss
# from multi_task_loss import MultiTaskLoss
from optimizers import Ranger
from schedulers import cyclical_lr
from apex import amp
sys.path.append('./11_Models/')
from model_190819 import Model

### others
import argparse
import os
from collections import defaultdict
import datetime
import numpy as np
import torch
from tqdm import tqdm
import time
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import gc
from sklearn.utils import shuffle
import socket
hostname = socket.gethostname()
if hostname == 'localhost.localdomain':
  from delogger import Delogger
  Delogger.is_debug_stream = True
  debuglog = Delogger.line_profiler


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# ap.add_argument("-dev", "--developping", action='store_true')
ap.add_argument("-tra", "--train_first", action='store_true')
ap.add_argument("-bi", "--batch_interval", default=100, type=int, help=" ")
ap.add_argument("-nl", "--n_load_all", default=186, type=int, help=" ")
ap.add_argument("-gclp", "--grad_clip_value", default=0, type=int, help=" ")
ap.add_argument("-wclp", "--weight_clip_value", default=0, type=int, help=" ")
ap.add_argument("-step", "--use_step_scheduler", action='store_true')
ap.add_argument("-clc", "--use_clc_scheduler", action='store_true')
ap.add_argument("-ilr", "--initial_learning_rate", default=3e-6, type=float, help=" ")
ap.add_argument("-bf", "--batch_factor", default=1, type=int, help=" ")
ap.add_argument("-nmt", "--n_mouse_tes", default="01", nargs='+', choices=["01", "02", "03", "04", "05"], help = " ")
ap.add_argument("-archi", "--use_archi", default="cnn", nargs='+', choices=["cnn", "rnn", "fft"], help = " ") # Feature extractors
ap.add_argument("-mtl", "--use_multitaskloss", action='store_true')
# About model architecture
ap.add_argument("-sr", "--samp_rate", default=1000, type=int, help=" ")
ap.add_argument("-msl", "--max_seq_len", default=1024, type=int, help=" ")

args = ap.parse_args()

args.use_fp16 = True
args.use_input_bn = True

def _print_log_tra(n_tra_counter, cum_loss_dur, cum_loss_lat, cum_loss_tot):
  avg_loss_dur = 1.0 * cum_loss_dur / n_tra_counter
  avg_loss_lat = 1.0 * cum_loss_lat / n_tra_counter
  avg_loss_tot = 1.0 * cum_loss_tot / n_tra_counter

  print('-----------------------------------------')
  print('\nTrain Epoch: {}: Avg. Loss_dur: {:.8f} Loss_lat: {:.8f} Loss_tot: {:.8f}'\
    .format(d['epoch'][-1], avg_loss_dur, avg_loss_lat, avg_loss_tot))
  print('-----------------------------------------')

  d['losses_dur_tra'].append(avg_loss_dur)
  d['losses_lat_tra'].append(avg_loss_lat)
  d['losses_tot_tra'].append(avg_loss_tot)
  d['lr'].append(mf.get_lr(optimizer) / p['batch_factor'])

def _save(i_batch):
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


def calc_normalization_params(N):
  print('Calculating parameters for normalization...')

  updater = mf.update_mean_var_std_arr()
  for _ in range(N):

    dl_tra = dl_fulf_tra.fulfill()
    for i_batch, batch in tqdm(enumerate(dl_tra)):
      _, _, Tb_lat = batch
      Tb_lat = Tb_lat.cpu().numpy()
      Tb_lat.astype(np.float64)
      log_lat_mean, _, log_lat_std = updater.update(np.log(Tb_lat + 1e-5))
  p['log(lat+1e-5)_mean'] = log_lat_mean
  p['log(lat+1e-5)_std'] = log_lat_std[0]


# @Delogger.line_memory_profiler
def train():
  timer('Epoch {} Train starts'.format(d['epoch'][-1]))
  model.train()
  if args.use_multitaskloss:
    multitaskloss.train()
  cum_loss_dur = 0
  cum_loss_lat = 0
  cum_loss_tot = 0
  n_tra_counter = 0
  dl_tra = dl_fulf_tra.fulfill()
  print('Current Learning Rate / Batch Factor: {}'.format(mf.get_lr(optimizer) / p['batch_factor'])) # scheduler.get_lr
  for i_batch, batch in tqdm(enumerate(dl_tra)):
    # Target
    Xb, Tb_dur, Tb_lat = batch
    Xb, Tb_dur, Tb_lat = Xb.to(m['device']), Tb_dur.to(m['device']), Tb_lat.to(m['device'])
    Tb_lat_logn = (torch.log(Tb_lat+1e-5)-p['log(lat+1e-5)_mean']) / p['log(lat+1e-5)_std'] # zscore
    #################

    optimizer.zero_grad()

    # Model Outputs
    pred_dur_mu, pred_dur_sigma, pred_lat_logn_mu, pred_lat_logn_sigma, _ = model(Xb)

    # losses
    loss_dur = pdf_loss(Tb_dur.float(), pred_dur_mu.float(), pred_dur_sigma.float()).mean()
    loss_lat = pdf_loss(Tb_lat.float(), pred_lat_logn_mu.float(), pred_lat_logn_sigma.float()).mean()

    if args.use_multitaskloss:
      losses = torch.stack([loss_dur, loss_lat])
      loss_tot = (multitaskloss(losses)).sum()
      with amp.scale_loss(loss_tot, optimizer) as scaled_loss_tot:
        scaled_loss_tot.backward()
    else:
      loss_tot = loss_dur + loss_lat
      with amp.scale_loss(loss_dur, optimizer) as scaled_loss_dur:
        scaled_loss_dur.backward(retain_graph=True)
      with amp.scale_loss(loss_lat, optimizer) as scaled_loss_lat:
        scaled_loss_lat.backward()

    if args.grad_clip_value:
      torch.nn.utils.clip_grad_norm_(model.parameters(), m['grad_clip_value'])

    optimizer.step()

    if args.weight_clip_value:
      for param in model.parameters():
        param.data.clamp_(-m['weight_clip_value'], m['weight_clip_value'])

    cum_loss_dur += loss_dur.detach().cpu().numpy().astype(np.float)
    cum_loss_lat += loss_lat.detach().cpu().numpy().astype(np.float)
    cum_loss_tot += loss_tot.detach().cpu().numpy().astype(np.float)
    n_tra_counter += len(Xb)

    if i_batch % p['log_batch_interval'] == 0:
      _print_log_tra(n_tra_counter, cum_loss_dur, cum_loss_lat, cum_loss_tot)

    if i_batch % p['save_batch_interval'] == 0:
      _save(i_batch)
      cum_loss_dur = 0
      cum_loss_lat = 0
      cum_loss_tot = 0
      n_tra_counter = 0

    if args.use_clc_scheduler:
      # if i_batch % p['lr_scheduler_batch_interval'] == 0:
      scheduler.step()
      print('Current Learning Rate / Batch Factor: {}'.format(mf.get_lr(optimizer) / p['batch_factor'])) # scheduler.get_lr
      # print('scheduler says: {}'.format(scheduler.get_lr()[-1] / p['batch_factor']))

  if args.use_step_scheduler:
      scheduler.step()
      print('Current Learning Rate / Batch Factor: {}'.format(mf.get_lr(optimizer) / p['batch_factor'])) # scheduler.get_lr
      # print('scheduler says: {}'.format(scheduler.get_lr()[-1] / p['batch_factor']))

  if d['epoch'][-1] == 1:
    d['n_tra'] = n_tra_counter

  del dl_tra, batch, Xb, Tb_dur, Tb_lat, Tb_lat_logn; gc.collect()

  timer('Epoch {} Train ends'.format(d['epoch'][-1]))


def _print_log_tes(n_tes_counter, cum_loss_dur, cum_loss_lat, cum_loss_tot):
  avg_loss_dur = 1.0 * cum_loss_dur / n_tes_counter
  avg_loss_lat = 1.0 * cum_loss_lat / n_tes_counter
  avg_loss_tot = 1.0 * cum_loss_tot / n_tes_counter

  d['losses_dur_tes'].append(avg_loss_dur)
  d['losses_lat_tes'].append(avg_loss_lat)
  d['losses_tot_tes'].append(avg_loss_tot)

  print('-----------------------------------------')
  print('\nTest Epoch: {}: Avg. Loss_dur: {:.8f}, Loss_lat: {:.8f}, Loss_tot: {:.8f}'\
        .format(d['epoch'][-1], avg_loss_dur, avg_loss_lat, avg_loss_tot))
  print('-----------------------------------------')


def test():
  timer('Epoch {} Test starts'.format(d['epoch'][-1]))
  dl_tes = dl_fulf_tes.fulfill()
  model.eval()
  if args.use_multitaskloss:
    multitaskloss.eval()
  p['n_tes'] = len(dl_tes.dataset)
  cum_loss_dur = 0
  cum_loss_lat = 0
  cum_loss_tot = 0
  n_tes_counter = 0
  with torch.no_grad():
    for i_batch, batch in enumerate(dl_tes):
      # Targets
      Xb, Tb_dur, Tb_lat = batch
      Xb, Tb_dur, Tb_lat = Xb.to(m['device']), Tb_dur.to(m['device']), Tb_lat.to(m['device'])
      Tb_lat_logn = (torch.log(Tb_lat+1e-5)-p['log(lat+1e-5)_mean']) / p['log(lat+1e-5)_std']

      # Model Outputs
      pred_dur_mu, pred_dur_sigma, pred_lat_logn_mu, pred_lat_logn_sigma, _ = model(Xb)

      # losses
      loss_dur = pdf_loss(Tb_dur.float(), pred_dur_mu.float(), pred_dur_sigma.float()).mean()
      loss_lat = pdf_loss(Tb_lat.float(), pred_lat_logn_mu.float(), pred_lat_logn_sigma.float()).mean()

      if args.use_multitaskloss:
        losses = torch.stack([loss_dur, loss_lat])
        loss_tot = (multitaskloss(losses)).sum()
      else:
        loss_tot = loss_dur + loss_lat

      cum_loss_dur += loss_dur.cpu().numpy().astype(np.float)
      cum_loss_lat += loss_lat.cpu().numpy().astype(np.float)
      cum_loss_tot += loss_tot.cpu().numpy().astype(np.float)
      n_tes_counter += len(Xb)

  del dl_tes, batch, Xb, Tb_dur, Tb_lat, Tb_lat_logn; gc.collect()

  _print_log_tes(n_tes_counter, cum_loss_dur, cum_loss_lat, cum_loss_tot)

  timer('Epoch {} Test ends'.format(d['epoch'][-1]))


def main():
  for epoch in range(1, p['max_epochs']+1):
    d['epoch'].append(epoch)
    if args.train_first:
      train()
    if epoch % p['test_epoch_interval'] == 0:
      test()
    if not args.train_first:
      train()
    d['time_by_epoch'].append(timer.from_prev_hhmmss)


def define_parameters():
  global ts, d, p, m, timer

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
  print('n_gpus : {}'.format(m['n_gpus']))

  ## Load Paths
  loadpath_npy_list = '../data/2kHz_npy_list.pkl'
  p['n_load_all'] = args.n_load_all # 186   # 12 -> fast, 176 -> w/o 05/day5, 186 -> full
  print('n_load_all : {}'.format(p['n_load_all']))
  fpaths = mf.pkl_load(loadpath_npy_list)[:p['n_load_all']]
  p['tes_keyword'] = args.n_mouse_tes # '02'
  print('Test Keyword: {}'.format(p['tes_keyword']))
  p['fpaths_tra'], p['fpaths_tes'] = mf.split_fpaths(fpaths, tes_keyword=p['tes_keyword'])

  ## Parameters
  p['samp_rate'] = args.samp_rate
  p['max_seq_len'] = args.max_seq_len
  p['use_perturb_tra'] = True
  p['use_perturb_tes'] = False
  p['batch_factor'] = args.batch_factor
  p['bs_tra'] = 64 * m['n_gpus'] * p['batch_factor']
  p['bs_tes'] = 64 * m['n_gpus'] * p['batch_factor']
  print('Batchsize: Train {}, Test {}'.format(p['bs_tra'], p['bs_tes']))

  # Intervals
  p['max_epochs'] = 10
  p['test_epoch_interval'] = 1
  p['save_batch_interval'] = args.batch_interval
  p['log_batch_interval'] = args.batch_interval
  p['lr_scheduler_batch_interval'] = args.batch_interval

  # NN
  # General
  m['init_lr'] = args.initial_learning_rate * p['batch_factor']
  m['grad_clip_value'] = args.grad_clip_value
  m['weight_clip_value'] = args.weight_clip_value
  # Save switches
  m['use_fp16'] = args.use_fp16
  m['use_multitaskloss'] = args.use_multitaskloss
  m['use_step_scheduler'] = args.use_step_scheduler
  m['use_clc_scheduler'] = args.use_clc_scheduler
  # Architecture
  m['n_features'] = 1
  m['hidden_size'] = 64
  m['num_layers'] = 4
  m['dropout_rnn'] = 0.1
  m['dropout_fc'] = 0.5
  m['bidirectional'] = True
  m['use_input_bn'] = args.use_input_bn
  m['rnn_archi'] = 'lstm'
  m['transformer_d_model'] = 128
  m['use_archi'] = args.use_archi

def init_ripple_detector_NN():
  global model, multitaskloss, optimizer, scheduler

  model = Model(input_size=m['n_features'],
                     max_seq_len=p['max_seq_len'],
                     samp_rate=p['samp_rate'],
                     hidden_size=m['hidden_size'],
                     num_layers=m['num_layers'],
                     dropout_rnn=m['dropout_rnn'],
                     dropout_fc=m['dropout_fc'],
                     bidirectional=m['bidirectional'],
                     use_input_bn=m['use_input_bn'],
                     use_rnn=('rnn' in args.use_archi),
                     use_cnn=('cnn' in args.use_archi),
                     use_transformer=('att' in args.use_archi),
                     transformer_d_model = m['transformer_d_model'],
                     use_fft=('fft' in args.use_archi),
                     use_wavelet_scat=('wlt' in args.use_archi),
                     rnn_archi=m['rnn_archi'],
                     ).to(m['device'])

  if args.use_multitaskloss:
    m['is_regression'] = torch.Tensor([True, True])
    multitaskloss = MultiTaskLoss(m['is_regression'])
    multitaskloss.to(m['device'])
    learnable_params = list(model.parameters()) + list(multitaskloss.parameters())
    num_losses = 1
  else:
    learnable_params = model.parameters()
    num_losses = 2

  if args.use_fp16:
    adam_eps = 1e-4
    optimizer = Ranger(learnable_params, lr = m['init_lr'], eps=adam_eps)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=num_losses)
  else:
    adam_eps = 1e-8
    optimizer = Ranger(learnable_params, lr = m['init_lr'], eps=adam_eps)

  if m['n_gpus'] > 1:
    model = torch.nn.DataParallel(model).to(m['device'])

  if args.use_step_scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10.0)

  if args.use_clc_scheduler:
    step_epochs = 2
    step_size = (n_tra / p['bs_tra']) * step_epochs # 100
    clc = cyclical_lr(step_size, min_lr=1.0, max_lr=10.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clc])


def init_dataloaders():
  global dl_fulf_tra, dl_fulf_tes

  dl_fulf_tra = dataloader_fulfiller(p['fpaths_tra'],
                                     p['samp_rate'],
                                     p['max_seq_len'],
                                     use_perturb=p['use_perturb_tra'],
                                     bs=p['bs_tra'],
                                     use_fp16=m['use_fp16'],
                                     istrain=True,
                                     use_shuffle=True)
  n_tra = dl_fulf_tra.get_n_samples()

  dl_fulf_tes = dataloader_fulfiller(p['fpaths_tes'],
                                     p['samp_rate'],
                                     p['max_seq_len'],
                                     use_perturb=p['use_perturb_tes'],
                                     bs=p['bs_tes'],
                                     use_fp16=m['use_fp16'],
                                     istrain=False,
                                     use_shuffle=False)


if __name__ == "__main__":
  define_parameters()
  init_ripple_detector_NN()
  init_dataloaders()
  calc_normalization_params(3)
  main()
