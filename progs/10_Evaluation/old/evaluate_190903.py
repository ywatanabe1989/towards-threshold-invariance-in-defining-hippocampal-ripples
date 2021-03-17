### my own packages
import sys
sys.path.append('./')
sys.path.append('./utils')
import myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_190829 import dataloader_fulfiller, load_lfps_rips_sec_sampr
sys.path.append('./07_Learning/')
from train_190829 import test, _print_log_tes
from pdf_loss import pdf_loss
from multi_task_loss import MultiTaskLoss
from optimizers import Ranger
from schedulers import cyclical_lr
from apex import amp
sys.path.append('./11_Models/')
from Model_190819 import Model

### others
import argparse
import os
from collections import defaultdict
import datetime
import math
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
from skimage import util


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-cnn", "--use_cnn", action='store_true')
ap.add_argument("-rnn", "--use_rnn", action='store_true')
ap.add_argument("-tf", "--use_transformer", action='store_true')
ap.add_argument("-fft", "--use_fft", action='store_true')
args = ap.parse_args()


## Load
if args.cnn:
  datadir = '../results/bests_190903/cnn/190830/034240/epoch_3/batch_100/'
if args.rnn:
  datadir = '../results/bests_190903/rnn/190830/090311/epoch_3/batch_900/'
if args.tf:
  datadir = '../results/bests_190903/attention/190902/053924/epoch_4/batch_100/'
if args.fft:
  datadir =  '../results/bests_190903/fft/190830/111458/epoch_3/batch_900/'

d = mf.pkl_load(datadir + 'data.pkl')
p = mf.pkl_load(datadir + 'params.pkl')
m = mf.pkl_load(datadir + 'model.pkl')
weight_path = datadir + 'weight.pth'
lfps, rips_sec, samp_rate = load_lfps_rips_sec_sampr(p['fpaths_tes'], lsampr=p['samp_rate'], use_fp16=m['use_fp16'], use_shuffle=False)

timer = mf.time_tracker()
## Initialize Neural Network
model = Model(input_size=m['n_features'],
                   max_seq_len=p['max_seq_len'],
                   samp_rate=p['samp_rate'],
                   hidden_size=m['hidden_size'],
                   num_layers=m['num_layers'],
                   dropout_rnn=m['dropout_rnn'],
                   dropout_fc=m['dropout_fc'],
                   bidirectional=m['bidirectional'],
                   use_input_bn=m['use_input_bn'],
                   use_rnn=m['use_rnn'],
                   use_cnn=m['use_cnn'],
                   use_transformer=m['use_transformer'],
                   transformer_d_model = m['transformer_d_model'],
                   use_fft=m['use_fft'],
                   use_wavelet_scat=m['use_wavelet_scat'],
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

model_state_dict = torch.load(weight_path)
model.load_state_dict(model_state_dict)
model.eval()
multitaskloss.eval()


# ## Prepair
# df = pd.DataFrame()
# df['lfp'] = lfp
# df['rip_start'] = 0
# rip_sec['start'] = (rip_sec['start_time']*p['samp_rate']).astype(int)
# df['rip_start'].iloc[rip_sec['start']] = 1

torch.backends.cudnn.enabled = True

# Predict Probability density fucntions
for i_lfp in range(len(lfps)):
  lfp, rip_sec = lfps[i_lfp], rips_sec[i_lfp]

  slices = util.view_as_windows(lfp, window_shape=(p['max_seq_len'],), step=1)
  preds_dur_mu = []
  preds_dur_sigma = []
  preds_lat_logn_mu = []
  preds_lat_logn_sigma = []
  bs = p['bs_tes'] * 1
  n_batches = math.ceil(len(slices)/bs)
  for i in tqdm(range(n_batches)): # 24h for one lfp on WS
    with torch.no_grad():
      Xb = slices[i*bs:(i+1)*bs]
      Xb = torch.tensor(Xb).cuda().unsqueeze(-1)
      pred_dur_mu, pred_dur_sigma, pred_lat_logn_mu, pred_lat_logn_sigma, _ = model(Xb)
      preds_dur_mu.append(pred_dur_mu.detach().cpu().numpy())
      preds_dur_sigma.append(pred_dur_sigma.detach().cpu().numpy())
      preds_lat_logn_mu.append(pred_lat_logn_mu.detach().cpu().numpy())
      preds_lat_logn_sigma.append(pred_lat_logn_sigma.detach().cpu().numpy())

  preds_dur_mu = np.hstack(preds_dur_mu)
  preds_dur_sigma = np.hstack(preds_dur_sigma)
  preds_lat_logn_mu = np.hstack(preds_lat_logn_mu)
  preds_lat_logn_sigma = np.hstack(preds_lat_logn_sigma)

  from collections import defaultdict
  preds = defaultdict(list)
  preds.update({'preds_dur_mu':preds_dur_mu,
                'preds_dur_sigma':preds_dur_sigma,
                'preds_lat_logn_mu':preds_lat_logn_mu,
                'preds_lat_logn_sigma':preds_lat_logn_sigma
                })

  i_lfp = 0
  savepath = datadir + 'preds/lfp_{}.pkl'.format(i_lfp)
  mf.pkl_save(preds, savepath)

# Tb_lat_logn = (torch.log(Tb_lat+1e-5)-p['log(lat+1e-5)_mean']) / p['log(lat+1e-5)_std'] # zscore
