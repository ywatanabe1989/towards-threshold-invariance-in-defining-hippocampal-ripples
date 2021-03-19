# #!/bin/python
### my own packages
import sys
sys.path.append('.')
import utils.myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_190829 import dataloader_fulfiller, load_lfps_rips_sec_sampr, load_lfp_rip_sec_sampr_from_lpath_lfp

# ### others
import argparse
from bisect import bisect_left, bisect_right
import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
# plt.ticklabel_format(style = 'sci', scilimits=(1,1))
import pandas as pd
from tqdm import tqdm
import scipy
import random
import socket
hostname = socket.gethostname()
from delogger import Delogger
Delogger.is_debug_stream = True
debuglog = Delogger.line_profiler



ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-cnn", "--use_cnn", action='store_true')
ap.add_argument("-rnn", "--use_rnn", action='store_true')
ap.add_argument("-tf", "--use_transformer", action='store_true')
ap.add_argument("-fft", "--use_fft", action='store_true')
ap.add_argument("-bf", "--batch_factor", type=int, default=1, help=" ")
args = ap.parse_args()

## Functions
def remove_preds_from_keys(dic):
  keys = list(dic.keys())
  for k in keys:
    new_k = k.replace('preds_', '')
    dic[new_k] = dic.pop(k)
  return dic

def take_abs_of_sigmas(preds):
  preds['dur_sigma'] = abs(preds['dur_sigma'])
  preds['lat_logn_sigma'] = abs(preds['lat_logn_sigma'])
  return preds

def _get_rv_z(mu_x, sigma_x):
  mu_y = log_lat_std * mu_x + log_lat_mean
  sigma_y = log_lat_std * sigma_x
  rv_z = scipy.stats.lognorm(s=sigma_y, scale=np.exp(mu_y))
  return rv_z

def _get_mode_z(mu_x, sigma_x):
  mu_y = log_lat_std * mu_x + log_lat_mean
  sigma_y = log_lat_std * sigma_x
  mode_z = np.exp(mu_y - sigma_y**2)
  return mode_z

def to_ci95(preds):
  ## Latency
  preds['lat_mode'] = _get_mode_z(preds['lat_logn_mu'], preds['lat_logn_sigma'])
  rv_z = _get_rv_z(preds['lat_logn_mu'], preds['lat_logn_sigma'])
  preds['lat_ppf_0.025'],  preds['lat_median'],  preds['lat_ppf_0.975'] = \
                                           rv_z.ppf(np.array([0.025, 0.500, 0.975])[:, np.newaxis])
  # preds['lat_median'] = rv_z.median()
  # preds['lat_mean'] = rv_z.mean()
  ## Duration
  preds['dur_ppf_0.025'] = preds['dur_mu'] - 1.96 * preds['dur_sigma']
  preds['dur_median'] = preds['dur_mu']
  preds['dur_ppf_0.975'] = preds['dur_mu'] + 1.96 * preds['dur_sigma']
  return preds

def set_ground_truth_values_to_df():
  lat_gt = []
  dur_gt = []

  start = 0
  for i in range(len(rip_sec)):
    end = math.ceil(rip_sec.iloc[i]['start_time']*p['samp_rate']) # int
    embed_lat = np.arange(end - start, 0, -1)
    embed_dur = np.ones_like(embed_lat) * rip_sec.iloc[i]['duration']
    lat_gt.append(embed_lat)
    dur_gt.append(embed_dur)
    start = end

  lat_gt = np.hstack(lat_gt)
  dur_gt = np.hstack(dur_gt)
  nan_arr = np.ones(len(df) - len(lat_gt)) * np.nan

  df['lat_gt'] = np.concatenate((lat_gt, nan_arr), axis=0)
  df['dur_gt'] = np.concatenate((dur_gt, nan_arr), axis=0)


def set_preds_to_df(preds, archi_name='CNN'):
  nan_arr = np.ones(p['max_seq_len']-1) * np.nan

  set_keys = ['lat_mode', 'lat_median', 'lat_ppf_0.025', 'lat_ppf_0.975',
              'dur_ppf_0.025', 'dur_median', 'dur_ppf_0.975'] # 'lat_mean', 'lat_ppf_0.500',
  for k in set_keys:
    item = np.concatenate((nan_arr, preds[k]), axis=-1)
    df[archi_name + '_' + k] = item


## Plot
def str2legend(string):
  string = string.replace('raw', 'Raw LFP [uV]')
  string = string.replace('filtered', 'Ripple-bandpassed LFP [uV]')
  string = string.replace('ripple', 'Ripple Time')
  string = string.replace('_lat_', ' Latency ')
  string = string.replace('_dur_', ' Duration ')
  string = string.replace('mode', 'mode [sec]')
  string = string.replace('median', 'median [sec]')

  #                         # ' Log-Normalized Latency\'s '+ '$\hat{\mu}$' +' [a.u.]')
  # string = string.replace('lat_mode_sec', ' Latency\'s '+'$\hat{mode}$'+' [sec]')
  # string = string.replace('lat_logn_sigma', ' Log-Normalized Latency\'s ' +'$\hat{\sigma}$'+' [a.u.]')
  # string = string.replace('dur_mu_sec', ' Duration\'s '+'$\hat{\mu}$'+' [sec]')
  # string = string.replace('dur_sigma_sec', ' Duration\'s '+'$\hat{\sigma}$'+' [sec]')
  return string

# @Delogger.line_memory_profiler
def plt_on_wave(df, rip_sec, start_sec=0, end_sec=60, yscale=None, **kwargs):
  start_pts = int(start_sec * p['samp_rate'])
  end_pts = int(end_sec * p['samp_rate'])
  dur_pts = end_pts - start_pts

  # X axis
  if start_pts < df.index[0]:
      start_pts = int(df.index[0])
      end_pts = start_pts + dur_pts
      start_sec = start_pts / p['samp_rate']
      end_sec = end_pts / p['samp_rate']
  print('Start {} s, End {} s'.format(start_sec, end_sec))

  df_plt = df.loc[start_pts:end_pts] # .copy()
  x_sec = np.linspace(start_sec, end_sec, len(df_plt))

  # plot
  lfp_op = np.array(['raw', 'filtered', 'ripple'])[np.array(lfp_sw, dtype=np.bool)]
  pred_op = np.array(['lat', 'dur'])[np.array(pred_sw, dtype=np.bool)]
  archi_op = np.array(['CNN', 'RNN', 'ATT', 'FFT'])[np.array(archi_sw, dtype=np.bool)]

  ## LFPs
  if 'raw' in lfp_op:
    plt.plot(x_sec, df_plt['LFP'], alpha=0.3, label=str2legend('raw'))
  if 'filtered' in lfp_op:
    plt.plot(x_sec, df_plt['150-250Hz'], alpha=0.3, label=str2legend('filtered'))
  if 'ripple' in lfp_op:
    rip_sec_plt = rip_sec[start_sec < rip_sec['start_time']]
    rip_sec_plt = rip_sec_plt[rip_sec_plt['end_time'] < end_sec]
    r = 0
    for ripple in rip_sec_plt.itertuples():
      if r == 0:
        plt.axvspan(ripple.start_time, ripple.end_time, alpha=0.3, color='red', zorder=1000, label=str2legend('ripple'))
        r =+ 1
      else:
        plt.axvspan(ripple.start_time, ripple.end_time, alpha=0.3, color='red', zorder=1000)

  ## Latency
  if 'lat' in pred_op:
    plt.plot(x_sec, df_plt['lat_gt'], 'k--', alpha=0.3,
             label=str2legend('Ground Truth Latency [sec]')) # Ground truth

    for archi_name in archi_op:
      key_root = '{}_lat_'.format(archi_name)
      # plt.plot(x_sec, df_plt[key_root + 'mode'], 'k--', alpha=0.3,
      #          label=str2legend(key_root + 'mode')) # mode
      # plt.plot(x_sec, df_plt[key_root + 'mean'], 'k--', alpha=0.3,
      #          label=str2legend(key_root + 'mean')) # mean
      plt.plot(x_sec, df_plt[key_root + 'median'], alpha=0.3, # 'k-'
               label=str2legend(key_root + 'median')) # median
      lwr = df_plt[key_root + 'ppf_0.025'] # 0.025 percentile
      upr = df_plt[key_root + 'ppf_0.975'] # 0.975 percentile
      plt.fill_between(x_sec, upr, lwr, alpha=0.3, linewidth=0,
                       label=str2legend(key_root + '95% Credential Interval')) # color="#3690C0"

  ## Duration
  if 'dur' in pred_op:
    plt.plot(x_sec, df_plt['dur_gt'], 'k--', alpha=0.3,
             label=str2legend('Ground Truth Duration [sec]')) # Ground truth

    for archi_name in archi_op:
      key_root = '{}_dur_'.format(archi_name)
      plt.plot(x_sec, df_plt[key_root + 'median'], alpha=0.3, # 'k-'
               label=str2legend(key_root + 'median')) # median
      lwr = df_plt[key_root + 'ppf_0.025'] # 0.025 percentile
      upr = df_plt[key_root + 'ppf_0.975'] # 0.975 percentile
      plt.fill_between(x_sec, upr, lwr, alpha=0.3, linewidth=0,
                       label=str2legend(key_root + '95% Credential Interval')) # color="#3690C0"

  plt.xlabel('Time (s)')
  plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)

  if yscale is not None:
    plt.yscale(yscale)
    plt.ylabel('(Log Scale)')

  plt.show()



########## Load ##########
datadir = '../report/old/190910/results/bests_190903/cnn/190830/034240/epoch_3/batch_100/' # fixme, as representative

# d = mf.pkl_load(datadir + 'data.pkl')
p = mf.pkl_load(datadir + 'params.pkl') # fpaths_tes, samp_rate
m = mf.pkl_load(datadir + 'model.pkl') # use_fp16
# weight_path = datadir + 'weight.pth'

# lfps, rips_sec, samp_rate = load_lfps_rips_sec_sampr(p['fpaths_tes'], lsampr=p['samp_rate'], use_fp16=m['use_fp16'], use_shuffle=False)
lfp, rip_sec, samp_rate = load_lfp_rip_sec_sampr_from_lpath_lfp(p['fpaths_tes'][0], lsampr=p['samp_rate'], use_fp16=m['use_fp16'])

archi_names = ['CNN', 'RNN', 'ATT', 'FFT']
preds_fpaths = ['../report/old/190910/results/bests_190903/cnn/190830/034240/epoch_3/batch_100/preds/lfp_0.pkl',
                '../report/old/190910/results/bests_190903/rnn/190830/090311/epoch_3/batch_900/preds/lfp_0.pkl',
                '../report/old/190910/results/bests_190903/attention/190902/053924/epoch_4/batch_100/preds/lfp_0.pkl',
                '../report/old/190910/results/bests_190903/fft/190830/111458/epoch_3/batch_900/preds/lfp_0.pkl']

# Normalizing parameters
log_lat_mean = mf.pkl_load('07_Learning/log(lat)_mean_std.pkl')['log(lat+1e-5)_mean']
log_lat_std = mf.pkl_load('07_Learning/log(lat)_mean_std.pkl')['log(lat+1e-5)_std']

# Raw NN outputs
preds = []
for i in range(len(preds_fpaths)):
  fpath = preds_fpaths[i]
  pred_dict = mf.pkl_load(fpath)
  pred_dict = remove_preds_from_keys(pred_dict)
  pred_dict = take_abs_of_sigmas(pred_dict) # Fix minus values in sigmas
  pred_dict = to_ci95(pred_dict) # Calculate Confidence Interval from Predicted \mu_x and \sigma_x
  preds.append(pred_dict)
del pred_dict
##############################

########## Organize in DataFrame ##########
df = pd.DataFrame()
# Raw
df['LFP'] = lfp
# Bandpassed
lowcut, highcut = 150, 250
bp = mf.bandpass(lowcut=lowcut, highcut=highcut, fs=p['samp_rate'])
passed = bp.butter_bandpass_filter(lfp)
bp_key = '{}-{}Hz'.format(lowcut, highcut)
df[bp_key] = passed
# Ground Truth Latency and Duration
set_ground_truth_values_to_df()
# Predictions
for archi_name, pred in zip(archi_names, preds):
  set_preds_to_df(pred, archi_name=archi_name)



########## Plot ##########
lfp_sw = [0, 0, 0] # 'raw', 'filtered', 'ripple'
pred_sw = [0, 1] # 'lat', 'dur'
archi_sw = [0, 0, 0, 1] # 'cnn', 'rnn', 'att', 'fft'
start = random.randint(0, int(len(lfp) / p['samp_rate'])) # 46859090 / 1000 - 10
end = start + 60 #  24 * 3600
# plt.ylim([0, 1e10])
archi_sw = [0, 1, 0, 0] # 'cnn', 'rnn', 'att', 'fft'
plt_on_wave(df, rip_sec, start, end, lfp_sw=lfp_sw, pred_sw=pred_sw, archi_sw=archi_sw, yscale='log')
##################################################


# # df.filter(regex=('(CNN.*lat_mode.*|lfp)')).plot()


# start = random.randint(0, 500000)
# end = start + 60

# start = 46859090 / 1000 - 10
# end = start + 60
# start = 0
# end = 24 * 3600
# keys=['ripple', 'CNN_lat_mode_sec', 'RNN_lat_mode_sec', 'ATT_lat_mode_sec', 'FFT_lat_mode_sec']
# plt_on_wave(df, rip_sec, start, end, keys=keys, yscale='log')
# plt_on_wave(df, rip_sec, start, end, keys=['ripple'])

# keys=['ripple', 'CNN_lat_mode_sec', 'CNN_lat_logn_sigma', 'CNN_dur_mu_sec', 'CNN_dur_sigma_sec']
# plt_on_wave(df, rip_sec, start, end, keys=keys, yscale='log')

# keys=['ripple', 'CNN_lat_logn_sigma', 'RNN_lat_logn_sigma', 'ATT_lat_logn_sigma', 'FFT_lat_logn_sigma']
# plt_on_wave(df, rip_sec, start, end, keys=keys)

# keys=['ripple', 'CNN_dur_mu_sec', 'RNN_dur_mu_sec', 'ATT_dur_mu_sec', 'FFT_dur_mu_sec']
# plt_on_wave(df, rip_sec, start, end, keys=keys)

# keys=['ripple', 'CNN_dur_sigma_sec', 'RNN_dur_sigma_sec', 'ATT_dur_sigma_sec', 'FFT_dur_sigma_sec']
# plt_on_wave(df, rip_sec, start, end, keys=keys)





# ## Prepair
# mu = 25
# sigma = 2
# _y = normal_pdf(x, mu, sigma)
# plt.plot(x, _y, label='$\mu$: {} $\sigma$: {}'.format(mu, sigma))
# plt.legend()
# plt.show()
# auc = _y.sum() / 1000

# y = transformed_pdf(x, mu, sigma)
# plt.plot(x, y, label='$\mu$: {} $\sigma$: {}'.format(mu, sigma))
# plt.legend()
# plt.show()
# auc = y.sum() / 1000

# def calc():
#   cum_predicted_pdf = np.zeros_like(lfp)
#   eval_len = int(1e7)
#   for i in tqdm(range(len(preds_cnn['lat_logn_mu']))):
#     input_last = p['max_seq_len'] + i
#     x = np.arange(0, len(lfp) - input_last)#[:eval_len]
#     x = x / p['samp_rate']
#     # y = transformed_pdf(x, preds_cnn['lat_logn_mu'][i], preds_cnn['lat_logn_sigma'][i])

#     y = transformed_pdf(x, preds_rnn['lat_logn_mu'][i], preds_rnn['lat_logn_sigma'][i])
#     print('Mode: {}'.format(y.max()))
#     print('Sum: {}'.format(y.sum()))
#     # plt.plot(x, y)
#     # plt.xlabel('Time [s]')
#     # plt.show()
#     # plt.plot(y)
#     cum_predicted_pdf[input_last:input_last+eval_len] += y
#     # df['predicted_pdf'] += y


# # def print_preds(preds, start, end):
# #   fig, ax = plt.subplots(6,1, sharex=True)
# #   x = np.arange(start, end)
# #   bias = 1024
# #   ax[0].plot(lfp[start:end], label='Raw LFP [uV]')
# #   ax[0].legend(loc='upper right')
# #   ax[1].plot(passed[start:end], label='Ripple Bandpassed LFP [uV]')
# #   ax[1].legend(loc='upper right')
# #   ax[2].plot(x, preds['lat_logn_mu'][start-bias:end-bias], label='lat_logn_mu')
# #   ax[2].legend(loc='upper right')
# #   ax[3].plot(x, preds['lat_logn_sigma'][start-bias:end-bias], label='lat_logn_sigma')
# #   ax[3].legend(loc='upper right')
# #   ax[4].plot(x, preds['dur_mu'][start-bias:end-bias], label='dur_mu')
# #   ax[4].legend(loc='upper right')
# #   ax[5].plot(x, preds['dur_sigma'][start-bias:end-bias], label='dur_sigma')
# #   ax[5].legend(loc='upper right')
# #   plt.show()

# # def print_preds_metrices(preds):
# #   print('Log-Normalized Latency \mu: [min:{}, max:{}]'\
# #         .format(min(preds['lat_logn_mu']), max(preds['lat_logn_mu'])))
# #   print('Log-Normalized Latency \sigma: [min:{}, max:{}]'\
# #         .format(min(preds['lat_logn_sigma']), max(preds['lat_logn_sigma'])))
# #   print('Duration \mu: [min:{}, max:{}]'\
# #         .format(min(preds['dur_mu']), max(preds['dur_mu'])))
# #   print('Duration \sigma: [min:{}, max:{}]'\
# #         .format(min(preds['dur_sigma']), max(preds['dur_sigma'])))
