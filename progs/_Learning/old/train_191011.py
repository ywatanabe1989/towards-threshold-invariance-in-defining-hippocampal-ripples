### my own packages
import sys
sys.path.append('./')
sys.path.append('./utils')
import myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_191009 import dataloader_fulfiller, cvt_Tb_distance_ms
sys.path.append('./07_Learning/')
from balance_xentropy_loss import BalanceCrossEntropyLoss
from ripple_detect_loss import get_lambda
# from pdf_loss import pdf_loss
from multi_task_loss import MultiTaskLoss
from optimizers import Ranger
from schedulers import cyclical_lr
from apex import amp
sys.path.append('./11_Models/')
from model_191015 import Model

### others
import argparse
from collections import defaultdict
import datetime
import gc
import numpy as np
import os
from sklearn.utils import shuffle
from scipy.stats import zscore
import torch
from tqdm import tqdm
import terminalplot as tp
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
# ap.add_argument("-dev", "--developping", action='store_true')
ap.add_argument("-l", "--use_loss", default='kldiv', choices=['xentropy', 'kldiv'], help = " ")
ap.add_argument("-nlb", "--apply_n_labels_balancing", action='store_true')
ap.add_argument("-da", "--apply_distance_adjustment", action='store_true')
ap.add_argument("-tra", "--train_first", action='store_true')
ap.add_argument("-sd", "--sd", default=5, choices=[2, 5], type=int, help = " ")
# ap.add_argument("-nl", "--n_load_all", default=12, type=int, help=" ") # 186
ap.add_argument("-nl", "--n_load_all", default=12, choices=[12, 186], type=int, help = " ")
ap.add_argument("-bi", "--batch_interval", default=100, type=int, help=" ")
ap.add_argument("-sch", "--use_scheduler", default="none", nargs='+', choices=["none", "step", "clc"], help = " ")
ap.add_argument("-ilr", "--initial_learning_rate", default=1e-3, type=float, help=" ")
ap.add_argument("-bf", "--batch_factor", default=10, type=int, help=" ")
ap.add_argument("-nmt", "--n_mouse_tes", default="01", nargs='+', choices=["01", "02", "03", "04", "05"], help = " ")
ap.add_argument("-ftr_ex", "--ftr_extractor", default="cnn", nargs='+', choices=["cnn", "rnn", "fft"], help = " ") # Feature extractors
ap.add_argument("-mtl", "--use_multitaskloss", action='store_true')
# About almost fixed values
ap.add_argument("-sr", "--samp_rate", default=1000, type=int, help=" ")
ap.add_argument("-msl", "--max_seq_len", default=200, type=int, help=" ")
ap.add_argument("-gclp", "--grad_clip_value", default=0, type=int, help=" ")
ap.add_argument("-wclp", "--weight_clip_value", default=0, type=int, help=" ")
args = ap.parse_args()

args.use_fp16 = True
args.use_input_bn = True
args.apply_n_labels_balancing = True

def _calc_ave(cum_dict):
  keys = cum_dict.keys()

  n_counter_keys = mf.search_str_list(keys, 'n_counter')
  n_counter = cum_dict[n_counter_keys[0]]

  cum_loss_keys = mf.search_str_list(keys, 'cum_loss_')

  for k in cum_loss_keys:
    cum_dict['ave_'+k.replace('cum_', '')] = 1.0 * cum_dict[k] / n_counter

  correct_keys = mf.search_str_list(keys, 'correct_')
  for k in correct_keys:
    cum_dict[k.replace('correct', 'accuracy')] = 1.0 * cum_dict[k] / n_counter

  return cum_dict


def _print_log_tra(cum_dict):
  print('-----------------------------------------')
  print('\nTrain Epoch: {}'.format(d['epoch'][-1]))
  pprint(cum_dict)
  print('-----------------------------------------')


def _save_tra(i_batch, cum_dict):
  keys = cum_dict.keys()
  loss_keys = mf.search_str_list(keys, 'loss')
  for k in loss_keys:
    d[k].append(cum_dict[k])

  d['lr'].append(mf.get_lr(optimizer) / p['batch_factor'])

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




# @Delogger.line_memory_profiler
def train():
  d['epoch'] = [1] if not d.get('epoch') else d.get('epoch')
  timer('Epoch {} Train starts'.format(d['epoch'][-1]))
  model.train()
  _ = multitaskloss.train() if m['use_multitaskloss'] else 0 # fixme
  keys_cum = ['cum_loss_isRipple_tra', 'cum_loss_tot_tra', 'correct_isRipple_tra', 'n_counter_tra',
              'cm_isRipple_tra']
  cum_dict = mf.init_dict(keys_cum)

  dl_tra = dl_fulf_tra.fulfill()
  print('Current Learning Rate / Batch Factor: {}'.format(mf.get_lr(optimizer) / p['batch_factor'])) # scheduler.get_lr

  for i_batch, batch in tqdm(enumerate(dl_tra)):
    # batch = next(iter(dl_tra)) if not 'batch' in locals() else batch # for developping purpose
    assert len(batch) == len(keys_to_pack)

    # Xb
    Xb_Tbs_dict = mf.init_dict(keys=keys_to_pack, values=batch) # Xb, Tb_dur, Tb_lat = batch
    Xb = Xb_Tbs_dict['Xb'].to(m['device']) # Xb, Tb_dur, Tb_lat = Xb.to(m['device']), Tb_dur.to(m['device']), Tb_lat.to(m['device'])

    # Tb
    Tb_isRipple = Xb_Tbs_dict['Tb_isRipple'].to(m['device'])
    Tb_distances_ms = Xb_Tbs_dict['Tb_distances_ms'].to(m['device'])

    optimizer.zero_grad()

    # Model Outputs
    pred_isRipple_logits = model(Xb.unsqueeze(-1))

    # losses
    if m['use_loss'] == 'xentropy':
      pred_isRipple_probs = softmax(pred_isRipple_logits)
      pred_isRipple_cls = torch.argmax(pred_isRipple_probs, dim=-1)
      loss_isRipple = xentropy_criterion(pred_isRipple_logits, Tb_isRipple.to(torch.long))
      correct_isRipple = (pred_isRipple_cls == Tb_isRipple.to(torch.long))
      cum_dict['correct_isRipple_tra'] += correct_isRipple.sum().detach().cpu().numpy()
      cum_dict['cm_isRipple_tra'] = mf.calc_confusion_matrix(Tb_isRipple.cpu(), pred_isRipple_cls.cpu(), \
                                                             labels=['noRipple', 'Ripple'])

    if m['use_loss'] == 'kldiv':
      Tb_distances_ms_onehot = cvt_Tb_distance_ms(Tb_distances_ms.clone(), p['max_distance_ms'])
      pred_isRipple_logprobs = log_softmax(pred_isRipple_logits)
      pred_isRipple_probs = pred_isRipple_logprobs.exp()
      loss_isRipple = kldiv_criterion(pred_isRipple_logprobs, Tb_distances_ms_onehot.to(m['device'])).sum(dim=-1)

      '''
      n_max_plot = 10
      n_plot = 0
      for _ in range(1000):
        i = np.random.randint(p['bs_tra'])
        distance_ms = Tb_distances_ms[i]
        true_ripple = -p['max_distance_ms'] < distance_ms and distance_ms < p['max_distance_ms']
        if true_ripple:
          print(distance_ms)
          plt.bar(np.arange(-(p['max_distance_ms']+1), (p['max_distance_ms']+1)+1), pred_isRipple_probs[i].detach().cpu())
          plt.title('Distance {} [ms]'.format(distance_ms))
          plt.pause(4)
          plt.close()
          n_plot += 1
          if n_plot == n_max_plot:
            break
      '''

    if m['apply_n_labels_balancing']:
      loss_isRipple = balance_loss(loss_isRipple, Tb_isRipple) # Scaling wrt sample sizes

    if m['apply_distance_adjustament']:
      lam = get_lambda(abs(Xb_Tbs_dict['Tb_distances_ms']), max_distance_ms=p['max_distance_ms'])
      loss_isRipple = (lam.to(torch.float).to(m['device'])*loss_isRipple) # Scaling wrt distances

    # plt.scatter(Tb_distances_ms.cpu().detach().numpy(), loss_isRipple.cpu().detach().numpy())
    loss_isRipple = loss_isRipple.mean()


    if m['use_multitaskloss']:
      losses = torch.stack([loss_isRipple])
      loss_tot = (multitaskloss(losses)).sum()
      with amp.scale_loss(loss_tot, optimizer) as scaled_loss_tot:
        scaled_loss_tot.backward()
    else:
      loss_tot = loss_isRipple
      with amp.scale_loss(loss_isRipple, optimizer) as scaled_loss_isRipple:
        scaled_loss_isRipple.backward()

    if args.grad_clip_value:
      torch.nn.utils.clip_grad_norm_(model.parameters(), m['grad_clip_value'])

    optimizer.step()

    if args.weight_clip_value:
      for param in model.parameters():
        param.data.clamp_(-m['weight_clip_value'], m['weight_clip_value'])

    cum_dict['cum_loss_isRipple_tra'] += loss_isRipple.detach().cpu().numpy().astype(np.float)
    cum_dict['cum_loss_tot_tra'] += loss_tot.detach().cpu().numpy().astype(np.float)
    cum_dict['n_counter_tra'] += len(Xb)

    if i_batch % p['print_log_batch_interval'] == 0:
      cum_dit = _calc_ave(cum_dict)
      _print_log_tra(cum_dict)

    if i_batch % p['save_batch_interval'] == 0:
      _save_tra(i_batch, cum_dict)
      cum_dict = mf.init_dict(keys_cum)

    if m['use_scheduler'] == 'clc':
      scheduler.step()
      print('Current Learning Rate / Batch Factor: {}'.format(mf.get_lr(optimizer) / p['batch_factor'])) # scheduler.get_lr

  if m['use_scheduler'] == 'step':
      scheduler.step()
      print('Current Learning Rate / Batch Factor: {}'.format(mf.get_lr(optimizer) / p['batch_factor'])) # scheduler.get_lr

  if d['epoch'][-1] == 1:
    d['n_tra'] = cum_dict['n_counter_tra']

  del dl_tra, batch, Xb, Tb_isRipple, Xb_Tbs_dict; gc.collect()

  timer('Epoch {} Train ends'.format(d['epoch'][-1]))


def _print_log_tes(cum_dict):
  print('-----------------------------------------')
  print('\nTest Epoch: {}'.format(d['epoch'][-1]))
  pprint(cum_dict)
  print('-----------------------------------------')


def _save_tes(cum_dict):
  keys = cum_dict.keys()
  loss_keys = mf.search_str_list(keys, 'loss')

  for k in loss_keys:
    d[k].append(cum_dict[k])


def test():
  d['epoch'] = [1] if not d.get('epoch') else d.get('epoch')
  timer('Epoch {} Test starts'.format(d['epoch'][-1]))
  dl_tes = dl_fulf_tes.fulfill()
  model.eval()
  _ = multitaskloss.eval() if m['use_multitaskloss'] else 0
  p['n_tes'] = len(dl_tes.dataset)

  keys_cum = ['cum_loss_isRipple_tes', 'cum_loss_tot_tes', 'correct_isRipple_tes', 'n_counter_tes',
              'cm_isRipple_tes']
  cum_dict = mf.init_dict(keys_cum)

  with torch.no_grad():
    for i_batch, batch in enumerate(dl_tes):
      # batch = next(iter(dl_tes)) if not 'batch' in locals() else batch # for developping purpose
      assert len(batch) == len(keys_to_pack)

      # Xb
      Xb_Tbs_dict = mf.init_dict(keys=keys_to_pack, values=batch) # Xb, Tb_dur, Tb_lat = batch
      Xb = Xb_Tbs_dict['Xb'].to(m['device']) # Xb, Tb_dur, Tb_lat = Xb.to(m['device']), Tb_dur.to(m['device']), Tb_lat.to(m['device'])

      # Tb
      Tb_isRipple = Xb_Tbs_dict['Tb_isRipple'].to(m['device'])
      Tb_distances_ms = Xb_Tbs_dict['Tb_distances_ms'].to(m['device'])

      # Model Outputs
      pred_isRipple_logits = model(Xb.unsqueeze(-1))

      # losses
      if m['use_loss'] == 'xentropy':
        pred_isRipple_probs = softmax(pred_isRipple_logits)
        pred_isRipple_cls = torch.argmax(pred_isRipple_probs, dim=-1)
        loss_isRipple = xentropy_criterion(pred_isRipple_logits, Tb_isRipple.to(torch.long))
        correct_isRipple = (pred_isRipple_cls == Tb_isRipple.to(torch.long))
        cum_dict['correct_isRipple_tes'] += correct_isRipple.sum().detach().cpu().numpy()
        cum_dict['cm_isRipple_tes'] = mf.calc_confusion_matrix(Tb_isRipple.cpu(), pred_isRipple_cls.cpu(), \
                                                               labels=['noRipple', 'Ripple'])
        '''
        plt.scatter(Tb_distances_ms.cpu().detach().numpy(),
                    pred_isRipple_probs[:, 1].cpu().detach().numpy())
        plt.xlabel('Distance from Ripple Peak[ms]')
        plt.ylabel('Predicted Ripple Probability (Ripple-center score)')


        n_max_plot = 5
        n_plot = 0
        for _ in range(1000):
          i = np.random.randint(p['bs_tes'])
          distance_ms = Tb_distances_ms[i]
          predicted_prob = pred_isRipple_probs[i, 1].cpu().detach().numpy()
          true_ripple = -p['max_distance_ms'] < distance_ms and distance_ms < p['max_distance_ms']
          pred_ripple = predicted_prob > 0.5
          # if (not true_ripple) and pred_ripple:
          # if true_ripple and pred_ripple:
          if (not true_ripple) and pred_ripple:
            plt.plot(np.arange(-int(len(Xb[0])/2), int(len(Xb[0])/2)), Xb[i].cpu().detach().numpy())
            plt.xlabel('Input Time [msec]')
            plt.ylabel('Amplitude [uV]')
            plt.title('Distance {:.2f} ms, Predicted Probabilitiy {:.2f}'\
                       .format(distance_ms, predicted_prob))
            plt.pause(10)
            plt.close()
            n_plot += 1
            if n_plot == n_max_plot:
              break
        '''


      if m['use_loss'] == 'kldiv':
        Tb_distances_ms_onehot = cvt_Tb_distance_ms(Tb_distances_ms.clone(), p['max_distance_ms'])
        pred_isRipple_logprobs = log_softmax(pred_isRipple_logits)
        pred_isRipple_probs = pred_isRipple_logprobs.exp()
        loss_isRipple = kldiv_criterion(pred_isRipple_logprobs, Tb_distances_ms_onehot.to(m['device'])).sum(dim=-1)

        '''
        n_max_plot = 10
        n_plot = 0
        for _ in range(1000):
          i = np.random.randint(p['bs_tra'])
          distance_ms = Tb_distances_ms[i]
          true_ripple = -p['max_distance_ms'] < distance_ms and distance_ms < p['max_distance_ms']
          if not true_ripple:
            print(distance_ms)
            plt.bar(np.arange(-(p['max_distance_ms']+1), (p['max_distance_ms']+1)+1), pred_isRipple_probs[i].detach().cpu())
            plt.title('Distance {} [ms]'.format(distance_ms))
            plt.pause(4)
            plt.close()
            n_plot += 1
            if n_plot == n_max_plot:
              break
        '''

      if m['apply_n_labels_balancing']:
        loss_isRipple = balance_loss(loss_isRipple, Tb_isRipple) # Scaling wrt sample sizes

      if m['apply_distance_adjustament']:
        lam = get_lambda(abs(Xb_Tbs_dict['Tb_distances_ms']), max_distance_ms=p['max_distance_ms'])
        loss_isRipple = (lam.to(torch.float).to(m['device'])*loss_isRipple) # Scaling wrt distances

      # plt.scatter(Tb_distances_ms.cpu().detach().numpy(), loss_isRipple.cpu().detach().numpy())
      loss_isRipple = loss_isRipple.mean()

      if m['use_multitaskloss']:
        losses = torch.stack([loss_isRipple])
        loss_tot = (multitaskloss(losses)).sum()
      else:
        loss_tot = loss_isRipple

      cum_dict['cum_loss_isRipple_tes'] += loss_isRipple.cpu().detach().numpy().astype(np.float)
      cum_dict['cum_loss_tot_tes'] += loss_tot.cpu().detach().numpy().astype(np.float)
      cum_dict['n_counter_tes'] += len(Xb)

  del dl_tes, batch, Xb, Tb_isRipple, Xb_Tbs_dict; gc.collect()

  cum_dict = _calc_ave(cum_dict)
  _print_log_tes(cum_dict)
  _save_tes(cum_dict)

  timer('Epoch {} Test ends'.format(d['epoch'][-1]))


def main():
  for epoch in range(1, p['max_epochs']+1):
    d['epoch'].append(epoch)
    if args.train_first:
      train()
    # if epoch % p['test_epoch_interval'] == 0:
    #   test()
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
  loadpath_npy_list = '../data/1kHz_npy_list.pkl'
  p['n_load_all'] = args.n_load_all # 186   # 12 -> fast, 176 -> w/o 05/day5, 186 -> full
  print('n_load_all : {}'.format(p['n_load_all']))
  fpaths = mf.pkl_load(loadpath_npy_list)[:p['n_load_all']]
  p['tes_keyword'] = args.n_mouse_tes # '02'
  print('Test Keyword: {}'.format(p['tes_keyword']))
  p['fpaths_tra'], p['fpaths_tes'] = mf.split_fpaths(fpaths, tes_keyword=p['tes_keyword'])

  ## Parameters
  p['samp_rate'] = args.samp_rate
  p['sd'] = args.sd
  p['max_seq_len'] = args.max_seq_len
  p['max_distance_ms'] = int(p['max_seq_len']/2)
  p['use_perturb_tra'] = True
  p['use_perturb_tes'] = False
  p['batch_factor'] = args.batch_factor
  p['bs_tra'] = 64 * m['n_gpus'] * p['batch_factor']
  p['bs_tes'] = 64 * m['n_gpus'] * p['batch_factor']
  print('Batchsize: Train {}, Test {}'.format(p['bs_tra'], p['bs_tes']))

  # Intervals
  p['max_epochs'] = 100
  p['test_epoch_interval'] = 1
  p['save_batch_interval'] = args.batch_interval
  p['print_log_batch_interval'] = args.batch_interval
  p['lr_scheduler_batch_interval'] = args.batch_interval

  # NN
  # General
  m['init_lr'] = args.initial_learning_rate * p['batch_factor']
  m['grad_clip_value'] = args.grad_clip_value
  m['weight_clip_value'] = args.weight_clip_value
  # Save switches
  m['use_fp16'] = args.use_fp16
  m['use_multitaskloss'] = args.use_multitaskloss
  m['use_scheduler'] = args.use_scheduler
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
  m['ftr_extractor'] = args.ftr_extractor
  m['use_loss'] = args.use_loss
  m['apply_n_labels_balancing'] = args.apply_n_labels_balancing
  m['apply_distance_adjustament'] = args.apply_distance_adjustment

def init_dataloaders():
  global dl_fulf_tra, dl_fulf_tes, keys_to_pack, kwargs_dl_tra

  keys_to_pack_detects_ripples = ['Xb', 'Tb_distances_ms', 'Tb_isRipple']
  keys_to_pack_estimates_ripple_params = ['Xb', 'Tb_distances_ms', 'ripple_relat_peak_posi', 'ripple_ave_power',
                                          'ripple_peak_power', 'ripple_peak_frequency_hz', 'gamma_ave_power']
  kwargs_dl_tra = {'samp_rate':p['samp_rate'],
                'sd':p['sd'],
                'use_fp16':m['use_fp16'],
                'use_shuffle':True,
                'max_seq_len_pts':p['max_seq_len'],
                'step':None,
                'use_perturb':True,
                'max_distance_ms':p['max_distance_ms'],
                'detects_ripples':True,
                'estimates_ripple_params':False,
                'bs':p['bs_tra'],
                'nw':10,
                'pm':True,
                'drop_last':True,
                'collate_fn_class':None,
                'keys_to_pack_detects_ripples':keys_to_pack_detects_ripples,
                'keys_to_pack_estimates_ripple_params':keys_to_pack_estimates_ripple_params,
                }
  keys_to_pack = kwargs_dl_tra['keys_to_pack_detects_ripples'] if kwargs_dl_tra['detects_ripples'] \
                                                            else kwargs_dl_tra['keys_to_pack_estimates_ripple_params']

  dl_fulf_tra = dataloader_fulfiller(p['fpaths_tra'], **kwargs_dl_tra)

  kwargs_tes = kwargs_dl_tra.copy()
  kwargs_tes.update({'use_shuffle':False, 'use_perturb':False, 'bs':p['bs_tes']})

  dl_fulf_tes = dataloader_fulfiller(p['fpaths_tes'], **kwargs_tes)


def init_ripple_detector_NN():
  global model, multitaskloss, optimizer, scheduler, softmax, log_softmax, xentropy_criterion, kldiv_criterion, balance_loss

  output_size_isRipple = 2 if m['use_loss'] == 'xentropy' else p['max_seq_len']+3
  model = Model(input_size=m['n_features'],
                output_size_isRipple=output_size_isRipple,
                max_seq_len=p['max_seq_len'],
                samp_rate=p['samp_rate'],
                hidden_size=m['hidden_size'],
                num_layers=m['num_layers'],
                dropout_rnn=m['dropout_rnn'],
                dropout_fc=m['dropout_fc'],
                bidirectional=m['bidirectional'],
                use_input_bn=m['use_input_bn'],
                use_rnn=('rnn' in args.ftr_extractor),
                use_cnn=('cnn' in args.ftr_extractor),
                use_transformer=('att' in args.ftr_extractor),
                transformer_d_model = m['transformer_d_model'],
                use_fft=('fft' in args.ftr_extractor),
                use_wavelet_scat=('wlt' in args.ftr_extractor),
                rnn_archi=m['rnn_archi'],
                ).to(m['device'])

  if m['use_multitaskloss']:
    m['is_regression'] = torch.Tensor([True, True])
    multitaskloss = MultiTaskLoss(m['is_regression'])
    multitaskloss.to(m['device'])
    learnable_params = list(model.parameters()) + list(multitaskloss.parameters())
    num_losses = 1
  else:
    learnable_params = model.parameters()
    num_losses = 2

  if m['use_fp16']:
    adam_eps = 1e-4
    optimizer = Ranger(learnable_params, lr = m['init_lr'], eps=adam_eps)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=num_losses)
  else:
    adam_eps = 1e-8
    optimizer = Ranger(learnable_params, lr = m['init_lr'], eps=adam_eps)

  if m['n_gpus'] > 1:
    model = torch.nn.DataParallel(model).to(m['device'])

  if m['use_scheduler'] == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10.0)

  if m['use_scheduler'] == 'clc':
    step_epochs = 2
    step_size = (n_tra / p['bs_tra']) * step_epochs # 100
    clc = cyclical_lr(step_size, min_lr=1.0, max_lr=10.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clc])

  softmax = torch.nn.Softmax(dim=-1)
  log_softmax = torch.nn.LogSoftmax(dim=-1)
  xentropy_criterion = torch.nn.CrossEntropyLoss(reduction='none')
  kldiv_criterion = torch.nn.KLDivLoss(reduction='none')
  n_classes = 2
  balance_loss = BalanceCrossEntropyLoss(n_classes)

def load_parameters(datadir):
  global d, p, m, dl_fulf_tes
  # datadir = '../results/191014/075644/epoch_10/batch_29600/'
  d = mf.pkl_load(datadir + 'data.pkl')
  p = mf.pkl_load(datadir + 'params.pkl')
  m = mf.pkl_load(datadir + 'model.pkl')
  weight_path = datadir + 'weight.pth'
  model_state_dict = torch.load(weight_path)
  model.load_state_dict(model_state_dict)
  '''
  m['use_loss'] = 'xentropy'
  '''

  # kwargs_tes = kwargs_dl_tra.copy()
  # kwargs_tes.update({'use_shuffle':False, 'use_perturb':False, 'bs':p['bs_tes'], 'step':50})

  # dl_fulf_tes = dataloader_fulfiller(p['fpaths_tes'][:2], **kwargs_tes)


if __name__ == "__main__":
  define_parameters()
  init_dataloaders()
  init_ripple_detector_NN()
  # calc_normalization_params(3)
  # datadir = '../results/191014/075644/epoch_10/batch_29600/'
  # if datadir:
  #   load_parameters(datadir)
  main()
