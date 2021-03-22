# #!/bin/python
### my own packages
import sys
import utils.myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_190829 import dataloader_fulfiller, load_lfps_rips_sec_sampr, load_lfp_rip_sec_sampr_from_lpath_lfp

# ### others
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
# plt.ticklabel_format(style = 'sci', scilimits=(1,1))
import pandas as pd
from tqdm import tqdm


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-cnn", "--use_cnn", action='store_true')
ap.add_argument("-rnn", "--use_rnn", action='store_true')
ap.add_argument("-tf", "--use_transformer", action='store_true')
ap.add_argument("-fft", "--use_fft", action='store_true')
ap.add_argument("-bf", "--batch_factor", type=int, default=1, help=" ")
args = ap.parse_args()

## Functions
def to_abs(preds):
  preds['preds_dur_sigma'] = abs(preds['preds_dur_sigma'])
  preds['preds_lat_logn_sigma'] = abs(preds['preds_lat_logn_sigma'])
  return preds

def normal_pdf(x, mu, sigma):
  var = sigma**2
  return 1./((2*np.pi*var)**0.5) * np.exp(-(x-mu)**2 / (2*var))

def g(y): # x = g(y)
  x = ( np.log(y + 1e-5) - log_lat_mean )/ log_lat_std
  return x

def f(x): # y = f(x), f(x) = invg(x), x: logn_lat
  y = np.exp(log_lat_std*x + log_lat_mean) - 1e-5
  return y # y: lat_sec

def transformed_pdf(z, mu, sigma):
  y = z / 1000 # p['samp_rate']
  x = g(y)
  dxdy = 1 / (log_lat_std * (y + 1e-5))
  dydz = 1 / p['samp_rate']
  return normal_pdf(x, mu, sigma) * abs(dxdy) * abs(dydz)

def print_preds(preds, start, end):
  fig, ax = plt.subplots(6,1, sharex=True)
  x = np.arange(start, end)
  bias = 1024
  ax[0].plot(lfp[start:end], label='Raw LFP [uV]')
  ax[0].legend(loc='upper right')
  ax[1].plot(passed[start:end], label='Ripple Bandpassed LFP [uV]')
  ax[1].legend(loc='upper right')
  ax[2].plot(x, preds['preds_lat_logn_mu'][start-bias:end-bias], label='preds_lat_logn_mu')
  ax[2].legend(loc='upper right')
  ax[3].plot(x, preds['preds_lat_logn_sigma'][start-bias:end-bias], label='preds_lat_logn_sigma')
  ax[3].legend(loc='upper right')
  ax[4].plot(x, preds['preds_dur_mu'][start-bias:end-bias], label='preds_dur_mu')
  ax[4].legend(loc='upper right')
  ax[5].plot(x, preds['preds_dur_sigma'][start-bias:end-bias], label='preds_dur_sigma')
  ax[5].legend(loc='upper right')
  plt.show()

def print_preds_metrices(preds):
  print('Log-Normalized Latency \mu: [min:{}, max:{}]'\
        .format(min(preds['preds_lat_logn_mu']), max(preds['preds_lat_logn_mu'])))
  print('Log-Normalized Latency \sigma: [min:{}, max:{}]'\
        .format(min(preds['preds_lat_logn_sigma']), max(preds['preds_lat_logn_sigma'])))
  print('Duration \mu: [min:{}, max:{}]'\
        .format(min(preds['preds_dur_mu']), max(preds['preds_dur_mu'])))
  print('Duration \sigma: [min:{}, max:{}]'\
        .format(min(preds['preds_dur_sigma']), max(preds['preds_dur_sigma'])))

def set_preds(preds, archi='CNN'):
  nan_arr = np.ones(p['max_seq_len']-1) * np.nan

  lat_logn_mu = np.concatenate((nan_arr, preds['preds_lat_logn_mu']), axis=0)
  lat_mode_sec = f(lat_logn_mu)
  lat_logn_sigma = np.concatenate((nan_arr, preds['preds_lat_logn_sigma']), axis=0)
  dur_mu_sec = np.concatenate((nan_arr, preds['preds_dur_mu']), axis=0)
  dur_sigma_sec = np.concatenate((nan_arr, preds['preds_dur_sigma']), axis=0)

  df['{}_lat_logn_mu'.format(archi)] = lat_logn_mu
  df['{}_lat_mode_sec'.format(archi)] = lat_mode_sec
  df['{}_lat_logn_sigma'.format(archi)] = lat_logn_sigma
  df['{}_dur_mu_sec'.format(archi)] = dur_mu_sec
  df['{}_dur_sigma_sec'.format(archi)] = dur_sigma_sec

def key2legendtxt(k):
  k = k.replace('lfp', 'Raw LFP [uV]')
  k = k.replace('Attention', 'ATT')
  k = k.replace('_lat_logn_mu', ' Log-Normalized Latency\'s '+ '$\hat{\mu}$' +' [a.u.]')
  k = k.replace('_lat_mode_sec', ' Latency\'s '+'$\hat{mode}$'+' [sec]')
  k = k.replace('_lat_logn_sigma', ' Log-Normalized Latency\'s ' +'$\hat{\sigma}$'+' [a.u.]')
  k = k.replace('_dur_mu_sec', ' Duration\'s '+'$\hat{\mu}$'+' [sec]')
  k = k.replace('_dur_sigma_sec', ' Duration\'s '+'$\hat{\sigma}$'+' [sec]')
  return k

def plt_on_wave(df, rip_sec, start_sec, end_sec, keys=['raw', 'bandpassed', 'ripple', 'CNN_lat_mu_sec'], yscale=None):
  plt_start_sec = start_sec
  plt_end_sec = end_sec

  plt_start = int(plt_start_sec * p['samp_rate'])
  plt_end = int(plt_end_sec * p['samp_rate'])
  plt_dur = plt_end - plt_start

  # axis
  if plt_start < df.index[0]:
      plt_start = int(df.index[0])
      plt_end = plt_start + plt_dur
      plt_start_sec = plt_start / p['samp_rate']
      plt_end_sec = plt_end / p['samp_rate']
  print('Start {} s, End {} s'.format(plt_start_sec, plt_end_sec))

  df_plt = df.loc[plt_start:plt_end].copy()
  x_sec = np.linspace(plt_start_sec, plt_end_sec, len(df_plt))

  # plot
  handles = []
  for k in keys:
    if k == 'ripple':
      rip_sec_plt = rip_sec[plt_start_sec < rip_sec['start_time']]
      rip_sec_plt = rip_sec_plt[rip_sec_plt['end_time'] < plt_end_sec]
      r = 0
      for ripple in rip_sec_plt.itertuples():
        if r == 0:
          plt.axvspan(ripple.start_time, ripple.end_time, alpha=0.3, color='red', zorder=1000, label='Ripple Time')
          r =+ 1
        else:
          plt.axvspan(ripple.start_time, ripple.end_time, alpha=0.3, color='red', zorder=1000)
    else:
      plt.plot(x_sec, df_plt[k], alpha=0.3, label=key2legendtxt(k))

  # if 'preds_prob' in plots:
  #   plt.fill_between(x_sec, 0, df_plt['preds_prob']*factor, color='orange', alpha=0.3, label='preds_prob') # preds_prob

  plt.xlabel('Time (s)')
  # plt.ylabel('Amplitude (uV)')
  plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)

  if yscale is not None:
    plt.yscale(yscale)
    plt.ylabel('(Log Scale)')
    # plt.grid(True)# , which='minor')

  plt.show()
  # mf.pkl_save(fig, 'savepath.pkl') # https://yatt.hatenablog.jp/entry/20170813/1502599853


## Load
datadir = '../report/190910/results/bests_190903/cnn/190830/034240/epoch_3/batch_100/'
# if args.use_cnn:
#   datadir = '../report/190910/results/bests_190903/cnn/190830/034240/epoch_3/batch_100/'
# if args.use_rnn:
#   datadir = '../report/190910/results/bests_190903/rnn/190830/090311/epoch_3/batch_900/'
# if args.use_transformer:
#   datadir = '../report/190910/results/bests_190903/attention/190902/053924/epoch_4/batch_100/'
# if args.use_fft:
#   datadir =  '../report/190910/results/bests_190903/fft/190830/111458/epoch_3/batch_900/'

# d = mf.pkl_load(datadir + 'data.pkl')
p = mf.pkl_load(datadir + 'params.pkl') # fpaths_tes, samp_rate
m = mf.pkl_load(datadir + 'model.pkl') # use_fp16
# weight_path = datadir + 'weight.pth'

# lfps, rips_sec, samp_rate = load_lfps_rips_sec_sampr(p['fpaths_tes'], lsampr=p['samp_rate'], use_fp16=m['use_fp16'], use_shuffle=False)
lfp, rip_sec, samp_rate = load_lfp_rip_sec_sampr_from_lpath_lfp(p['fpaths_tes'][0], lsampr=p['samp_rate'], use_fp16=m['use_fp16'])

preds_cnn = mf.pkl_load('../report/190910/results/bests_190903/cnn/190830/034240/epoch_3/batch_100/preds/lfp_0.pkl')
preds_rnn = mf.pkl_load('../report/190910/results/bests_190903/rnn/190830/090311/epoch_3/batch_900/preds/lfp_0.pkl')
preds_tf = mf.pkl_load('../report/190910/results/bests_190903/attention/190902/053924/epoch_4/batch_100/preds/lfp_0.pkl')
preds_fft = mf.pkl_load('../report/190910/results/bests_190903/fft/190830/111458/epoch_3/batch_900/preds/lfp_0.pkl')

# Some fixed values
log_lat_mean = mf.pkl_load('07_Learning/log(lat)_mean_std.pkl')['log(lat+1e-5)_mean']
log_lat_std = mf.pkl_load('07_Learning/log(lat)_mean_std.pkl')['log(lat+1e-5)_std']

# Fix minus values in sigmas
preds_cnn = to_abs(preds_cnn)
preds_rnn = to_abs(preds_rnn)
preds_tf = to_abs(preds_tf)
preds_fft = to_abs(preds_fft)

## Organize in DataFrame
df = pd.DataFrame()
# raw
df['lfp'] = lfp
# bandpassed
lowcut, highcut = 150, 250
bp = mf.bandpass(lowcut=lowcut, highcut=highcut, fs=p['samp_rate'])
passed = bp.butter_bandpass_filter(lfp)
bp_key = '{}-{}Hz'.format(lowcut, highcut)
df[bp_key] = passed
# Predicted mu, sigma
set_preds(preds_cnn, archi='CNN')
set_preds(preds_rnn, archi='RNN')
set_preds(preds_tf, archi='Attention')
set_preds(preds_fft, archi='FFT')

# plots=['raw', 'bandpassed', 'ripple', 'CNN_lat_mu_sec', 'RNN_lat_mu_sec', 'Attention_lat_mu_sec', 'Attention_lat_mu_sec']
## Scaling to view
# _df = df.copy()
# _df['CNN_lat_logn_sigma'] = df['CNN_lat_logn_sigma']*10


keys=['raw', 'ripple']
keys=[bp_key, 'ripple', 'CNN_lat_mode_sec', 'CNN_lat_logn_sigma',]
keys=['ripple', 'CNN_lat_mode_sec', 'CNN_lat_logn_sigma',]
keys=['ripple','CNN_lat_logn_sigma',]


keys=['ripple', 'CNN_lat_mode_sec', 'RNN_lat_mode_sec', 'Attention_lat_mode_sec', 'FFT_lat_mode_sec']

start = 46859090 / 1000 - 10
end = start + 60
start = 0
end = 24 * 3600
keys=['ripple', 'CNN_lat_mode_sec', 'RNN_lat_mode_sec', 'Attention_lat_mode_sec', 'FFT_lat_mode_sec']
plt_on_wave(df, rip_sec, start, end, keys=keys, yscale='log')
plt_on_wave(df, rip_sec, start, end, keys=['ripple'])

keys=['ripple', 'CNN_lat_mode_sec', 'CNN_lat_logn_sigma', 'CNN_dur_mu_sec', 'CNN_dur_sigma_sec']
plt_on_wave(df, rip_sec, start, end, keys=keys, yscale='log')

keys=['ripple', 'CNN_lat_logn_sigma', 'RNN_lat_logn_sigma', 'Attention_lat_logn_sigma', 'FFT_lat_logn_sigma']
plt_on_wave(df, rip_sec, start, end, keys=keys)

keys=['ripple', 'CNN_dur_mu_sec', 'RNN_dur_mu_sec', 'Attention_dur_mu_sec', 'FFT_dur_mu_sec']
plt_on_wave(df, rip_sec, start, end, keys=keys)

keys=['ripple', 'CNN_dur_sigma_sec', 'RNN_dur_sigma_sec', 'Attention_dur_sigma_sec', 'FFT_dur_sigma_sec']
plt_on_wave(df, rip_sec, start, end, keys=keys)


import random
start = random.randint(0, 500000)
end = start + 60
plt_on_wave(df, rip_sec, start, end, keys=keys)

df['CNN_lat_mode_sec'][df['CNN_lat_mode_sec'] == df['CNN_lat_mode_sec'].min()] # 46859090
# start, end = 1024, 2024000
# print_preds(preds_cnn, start, end)


# print_preds_metrices(preds_cnn)
# print_preds_metrices(preds_rnn)
# print_preds_metrices(preds_tf)
# print_preds_metrices(preds_fft)

## Transform


# get ransformer probability density



## Plot
## Prepair
# # ripple
# df['rip_start'] = 0
# rip_sec['start'] = (rip_sec['start_time']*p['samp_rate']).astype(int)
# df['rip_start'].iloc[rip_sec['start']] = 1

# df['predicted_pdf'] = 0

mu = 25
sigma = 2
_y = normal_pdf(x, mu, sigma)
plt.plot(x, _y, label='$\mu$: {} $\sigma$: {}'.format(mu, sigma))
plt.legend()
plt.show()
auc = _y.sum() / 1000

y = transformed_pdf(x, mu, sigma)
plt.plot(x, y, label='$\mu$: {} $\sigma$: {}'.format(mu, sigma))
plt.legend()
plt.show()
auc = y.sum() / 1000

def calc():
  cum_predicted_pdf = np.zeros_like(lfp)
  eval_len = int(1e7)
  for i in tqdm(range(len(preds_cnn['preds_lat_logn_mu']))):
    input_last = p['max_seq_len'] + i
    x = np.arange(0, len(lfp) - input_last)#[:eval_len]
    x = x / p['samp_rate']
    # y = transformed_pdf(x, preds_cnn['preds_lat_logn_mu'][i], preds_cnn['preds_lat_logn_sigma'][i])

    y = transformed_pdf(x, preds_rnn['preds_lat_logn_mu'][i], preds_rnn['preds_lat_logn_sigma'][i])
    print('Mode: {}'.format(y.max()))
    print('Sum: {}'.format(y.sum()))
    # plt.plot(x, y)
    # plt.xlabel('Time [s]')
    # plt.show()
    # plt.plot(y)
    cum_predicted_pdf[input_last:input_last+eval_len] += y
    # df['predicted_pdf'] += y

# x = np.arange(len(lfp))
# plt_len = 10000
# plt.plot(x[:plt_len], y[:plt_len])
# plt.show()

# calc()

## Plot output on wave





















#

# timer = mf.time_tracker()
# ## Initialize Neural Network
# model = Model(input_size=m['n_features'],
#                    max_seq_len=p['max_seq_len'],
#                    samp_rate=p['samp_rate'],
#                    hidden_size=m['hidden_size'],
#                    num_layers=m['num_layers'],
#                    dropout_rnn=m['dropout_rnn'],
#                    dropout_fc=m['dropout_fc'],
#                    bidirectional=m['bidirectional'],
#                    use_input_bn=m['use_input_bn'],
#                    use_rnn=m['use_rnn'],
#                    use_cnn=m['use_cnn'],
#                    use_transformer=m['use_transformer'],
#                    transformer_d_model = m['transformer_d_model'],
#                    use_fft=m['use_fft'],
#                    use_wavelet_scat=m['use_wavelet_scat'],
#                    rnn_archi=m['rnn_archi'],
#                    ).to(m['device'])

# if m['use_multitaskloss']:
#   m['is_regression'] = torch.Tensor([True, True])
#   multitaskloss = MultiTaskLoss(m['is_regression'])
#   multitaskloss.to(m['device'])
#   learnable_params = list(model.parameters()) + list(multitaskloss.parameters())
#   num_losses = 1
# else:
#   learnable_params = model.parameters()
#   num_losses = 2

# if m['use_fp16']:
#   adam_eps = 1e-4
#   optimizer = Ranger(learnable_params, lr = m['init_lr'], eps=adam_eps)
#   model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=num_losses)
# else:
#   adam_eps = 1e-8
#   optimizer = Ranger(learnable_params, lr = m['init_lr'], eps=adam_eps)

# if m['n_gpus'] > 1:
#   model = torch.nn.DataParallel(model).to(m['device'])

# model_state_dict = torch.load(weight_path)
# model.load_state_dict(model_state_dict)
# model.eval()
# multitaskloss.eval()

# # ## Prepair
# # df = pd.DataFrame()
# # df['lfp'] = lfp
# # df['rip_start'] = 0
# # rip_sec['start'] = (rip_sec['start_time']*p['samp_rate']).astype(int)
# # df['rip_start'].iloc[rip_sec['start']] = 1

# torch.backends.cudnn.enabled = True

# # Predict Probability density fucntions
# for i_lfp in range(len(lfps)):
#   lfp, rip_sec = lfps[i_lfp], rips_sec[i_lfp]

#   slices = util.view_as_windows(lfp, window_shape=(p['max_seq_len'],), step=1)
#   preds_dur_mu = []
#   preds_dur_sigma = []
#   preds_lat_logn_mu = []
#   preds_lat_logn_sigma = []
#   bs = p['bs_tes'] * args.batch_factor
#   n_batches = math.ceil(len(slices)/bs)
#   for i in tqdm(range(n_batches)): # 24h for one lfp on WS
#     with torch.no_grad():
#       Xb = slices[i*bs:(i+1)*bs]
#       Xb = torch.tensor(Xb).cuda().unsqueeze(-1)
#       pred_dur_mu, pred_dur_sigma, pred_lat_logn_mu, pred_lat_logn_sigma, _ = model(Xb)
#       preds_dur_mu.append(pred_dur_mu.detach().cpu().numpy())
#       preds_dur_sigma.append(pred_dur_sigma.detach().cpu().numpy())
#       preds_lat_logn_mu.append(pred_lat_logn_mu.detach().cpu().numpy())
#       preds_lat_logn_sigma.append(pred_lat_logn_sigma.detach().cpu().numpy())

#   preds_dur_mu = np.hstack(preds_dur_mu)
#   preds_dur_sigma = np.hstack(preds_dur_sigma)
#   preds_lat_logn_mu = np.hstack(preds_lat_logn_mu)
#   preds_lat_logn_sigma = np.hstack(preds_lat_logn_sigma)

#   from collections import defaultdict
#   preds = defaultdict(list)
#   preds.update({'preds_dur_mu':preds_dur_mu,
#                 'preds_dur_sigma':preds_dur_sigma,
#                 'preds_lat_logn_mu':preds_lat_logn_mu,
#                 'preds_lat_logn_sigma':preds_lat_logn_sigma
#                 })

#   savepath = datadir + 'preds/lfp_{}.pkl'.format(i_lfp)
#   mf.pkl_save(preds, savepath)

# # Tb_lat_logn = (torch.log(Tb_lat+1e-5)-p['log(lat+1e-5)_mean']) / p['log(lat+1e-5)_std'] # zscore
