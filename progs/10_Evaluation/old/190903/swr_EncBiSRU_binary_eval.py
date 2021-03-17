import sys
sys.path.append('/mnt/nvme/proj/progs/06_File_IO')
sys.path.append('/mnt/nvme/proj/progs/07_Learning/')
sys.path.append('/mnt/nvme/proj/progs/10_Evaluation/')
sys.path.append('/mnt/nvme/proj/progs/')
import os

from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
import multiprocessing as mp
from models_pt import EncBiSRU_binary as Model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import time
import utils.myfunc as mf
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from pprint import pprint
import scikitplot as skplt
import pandas as pd
from skimage import util
import seaborn as sns


## Load
load_ts, load_epoch = '190730/155837', 10
lpath_model = '../results/{}/epoch{}_model.pth'.format(load_ts, load_epoch)
lpath_optim = '../results/{}/epoch{}_optimizer.pth'.format(load_ts, load_epoch)
lpath_savedict = '../results/{}/epoch{}_savedict.pkl'.format(load_ts, load_epoch)
d = mf.pkl_load(lpath_savedict)
# lfps, rips_sec, samp_rate = mf.load_lfps_and_rips(d['fpaths_tes'], samp_rate=d['samp_rate'])
lfp, rip_sec, samp_rate = mf.load_an_lfp_and_a_rip(d['fpaths_tes'][0], samp_rate=d['samp_rate']) # 87850712
# Load NN
model = Model(input_size=d['n_features'], hidden_size=d['hidden_size'], num_layers=d['num_layers'],
              dropout=d['dropout'], bidirectional=d['bidirectional'])
model = model.cuda()
model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=d['lr'])
model_state_dict = torch.load(lpath_model)
model.load_state_dict(model_state_dict)
optimizer_state_dict = torch.load(lpath_optim)
optimizer.load_state_dict(optimizer_state_dict)
criterion_mse = nn.MSELoss(reduction='none')
criterion_xentropy = nn.CrossEntropyLoss(reduction='none') # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single cla
model.eval()


## Prepare
df = pd.DataFrame()
df['lfp'] = lfp
lowcut, highcut = 150, 250
bp = mf.bandpass(lowcut=lowcut, highcut=highcut, fs=d['samp_rate'])
passed = bp.butter_bandpass_filter(lfp)
df['{}-{}Hz'.format(lowcut, highcut)] = passed

df['rip_start'] = 0
rip_sec['start'] = (rip_sec['start_time']*d['samp_rate']).astype(int)
df['rip_start'].iloc[rip_sec['start']] = 1
# last_rip_start = int(rip_sec['start_time'].iloc[-1]*d['samp_rate']) # 89850625


# Target class
isn_range_ms = 100
isn_range_sec = isn_range_ms / 1000
isn_range = int(isn_range_sec * d['samp_rate'])
df['T_cls'] = 0
for i in range(isn_range):
  df['T_cls'] += df['rip_start'].shift(-i-1)
df['T_cls'][:d['max_seq_len']-1] = np.nan
df['T_cls'][df['T_cls'] > 1.0] = 1
df['T_cls'] = df['T_cls'].astype(int)


# Predict Probabilities and Classes
slices = util.view_as_windows(lfp, window_shape=(d['max_seq_len'],), step=1)
preds_prob = []
bs = d['bs_tes'] * 32
n_batches = int(len(slices)/bs + 1) # fixme
for i in tqdm(range(n_batches)): # 2h for one lfp (24h)
  with torch.no_grad():
    Xb = slices[i*bs:(i+1)*bs]
    _, _, pred_prob_isn = model(torch.tensor(Xb).cuda().unsqueeze(-1))
    pred_prob_isn = pred_prob_isn.detach().cpu().numpy()
    preds_prob.append(pred_prob_isn)
preds_prob = np.vstack(preds_prob)
'''
mf.pkl_save(preds_prob, 'preds_prob.pkl')
preds_prob = mf.pkl_load('preds_prob.pkl')
'''
assert len(preds_prob) == len(slices)
preds_cls = preds_prob.argmax(axis=1).astype(np.int)

df['preds_prob'] = 0
df['preds_prob'][:d['max_seq_len']-1] = np.nan
df['preds_prob'][d['max_seq_len']-1:d['max_seq_len']+len(preds_prob)] = preds_prob[:,1]

df['preds_cls'] = 0
df['preds_cls'][:d['max_seq_len']-1] = np.nan
df['preds_cls'][d['max_seq_len']-1:d['max_seq_len']+len(preds_cls)] = preds_cls
df['preds_cls'] = df['preds_cls'].astype(int)

df = df.dropna()

df['cum_preds_prob'] = 0 # cum prob
for i in range(isn_range):
    df['cum_preds_prob'] += df['preds_prob'].shift(i)


## Classification Report
savedir = '../results/{}/'.format(load_ts)
spath_df = savedir + 'epoch{}_cls_rep.csv'.format(load_epoch)
cr_df = mf.classification_report(df['T_cls'], df['preds_cls'], spath=spath_df)

## Confusion Matrix
spath_cm = savedir + 'epoch{}_conf_mat'.format(load_epoch)
cm, _ = mf.plot_confusion_matrix(df['preds_cls'], df['T_cls'], classes=['No Ripple','Ripple'], spath=spath_cm)
spath_cm_norm = savedir + 'epoch{}_conf_mat_norm'.format(load_epoch)
cm_norm, _ = mf.plot_confusion_matrix(df['preds_cls'], df['T_cls'], classes=['No Ripple','Ripple'], normalize=True, spath=spath_cm_norm)

## ROC Curve
spath_roc = savedir + 'epoch{}_ROC_Curve.png'.format(load_epoch)
mf.plot_roc_curve(df['T_cls'], df['preds_prob'], spath=spath_roc)

## Learning Curve
spath_lc = savedir + 'epoch{}_Learning_Curve.png'.format(load_epoch)
mf.plot_learning_curve(d, spath_lc)

## Plot output on wave
def plt_on_wave(df, start_sec, end_sec, plots=['raw', 'bandpassed', 'ripple', 'preds_prob', 'preds_cls', 'cum_preds_prob'], factor=100):
  plt_start_sec = start_sec
  plt_end_sec = end_sec

  plt_start = int(plt_start_sec * d['samp_rate'])
  plt_end = int(plt_end_sec * d['samp_rate'])
  plt_dur = plt_end - plt_start

  # axis
  if plt_start < df.index[0]:
      plt_start = int(df.index[0])
      plt_end = plt_start + plt_dur
      plt_start_sec = plt_start / d['samp_rate']
      plt_end_sec = plt_end / d['samp_rate']
  print('Start {} s, End {} s'.format(plt_start_sec, plt_end_sec))

  df_plt = df.loc[plt_start:plt_end].copy()
  x_sec = np.linspace(plt_start_sec, plt_end_sec, len(df_plt))

  # plot
  if 'raw' in plots:
    plt.plot(x_sec, df_plt['lfp'], alpha=0.3, label='raw') # raw
  if 'bandpassed' in plots:
    plt.plot(x_sec, df_plt['150-250Hz'], alpha=0.3, label='150-250 Hz')
  if 'ripple' in plots:
    rip_sec_plt = rip_sec[plt_start_sec < rip_sec['start_time']] # ripple
    rip_sec_plt = rip_sec_plt[rip_sec_plt['end_time'] < plt_end_sec]
    for ripple in rip_sec_plt.itertuples():
      plt.axvspan(ripple.start_time, ripple.end_time, alpha=0.3, color='red', zorder=1000, label='ripple')
  if 'preds_prob' in plots:
    plt.fill_between(x_sec, 0, df_plt['preds_prob']*factor, color='orange', alpha=0.3, label='preds_prob') # preds_prob
  if 'cum_preds_prob' in plots:
    plt.fill_between(x_sec, 0, df_plt['cum_preds_prob']*factor/100, color='orange', alpha=0.3, label='cum_preds_prob') # cum
  if 'preds_cls' in plots:
    plt.fill_between(x_sec, 0, df_plt['preds_cls']*factor, color='purple', alpha=0.3, label='preds_cls') # cls
  if 'T_cls' in plots:
    plt.fill_between(x_sec, 0, df_plt['T_cls']*factor, color='green', alpha=0.3, label='True_cls') # cls

  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude (uV)')
  plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
  plt.show()

# plt_on_wave(df, 1000, 2000, plots=['raw', 'ripple', 'cum_preds_prob'])
plt_on_wave(df, 0, 1000, plots=['T_cls', 'preds_prob'], factor=1)
