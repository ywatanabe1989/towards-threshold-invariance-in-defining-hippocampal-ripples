import logging
import sys
sys.path.append('./')
sys.path.append('./06_File_IO')
sys.path.append('./07_Learning/')
import os

from bisect import bisect_left, bisect_right
from collections import defaultdict
# import datetime
# from models_pt import EncBiSRU_binary as Model
import gc
import multiprocessing as mp
import numpy as np
import torch
import torch.utils.data as utils
import utils.myfunc as mf
from sklearn.utils import shuffle
from skimage import util
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN

import socket
hostname = socket.gethostname()
if hostname == 'localhost.localdomain':
  ###
  from delogger import Delogger
  Delogger.is_debug_stream = True
  debuglog = Delogger.line_profiler
  ###


def pkl_load(fpath):
    import pickle
    with open(fpath, 'rb') as f: # 'r'
        obj = pickle.load(f)
        return obj

def get_sampr_from_str(str):
  if str.find('2kHz') >= 0:
    sampr = 2000
  if str.find('1kHz') >= 0:
    sampr = 1000
  if str.find('500Hz') >= 0:
    sampr = 500
  return sampr

def get_lpath_rip_from_lpath_lfp(lpath_lfp):
  sampr = get_sampr_from_str(lpath_lfp)
  if sampr == 2000:
    lpath_rip = lpath_lfp.replace('2kHz', '1kHz').replace('.npy', '_rip_sec.pkl')
  if sampr == 1000:
    lpath_rip = lpath_lfp.replace('1kHz', '1kHz').replace('.npy', '_rip_sec.pkl')
  if sampr == 500:
    lpath_rip = lpath_lfp.replace('500Hz', '1kHz').replace('.npy', '_rip_sec.pkl')
  return lpath_rip

def replace_lpath_lfp_wrt_lsampr_and_dtype(lpath_lfp_2kHz, lsampr, use_fp16):
  if use_fp16:
    dtype = np.float16
    dtype_str = 'fp16'
  else:
    dtype = np.float32
    dtype_str = 'fp32'

  if lsampr == 2000:
    lpath_lfp = lpath_lfp_2kHz.replace('2kHz', '1kHz').replace('.npy', '_{}.npy'.format(dtype_str))
  if lsampr == 1000:
    lpath_lfp = lpath_lfp_2kHz.replace('2kHz', '1kHz').replace('.npy', '_{}.npy'.format(dtype_str))
  if lsampr == 500:
    lpath_lfp = lpath_lfp_2kHz.replace('2kHz', '500Hz').replace('.npy', '_{}.npy'.format(dtype_str))
  return lpath_lfp

def load_lfp_rip_sec_sampr_from_lpath_lfp(lpath_lfp_2kHz, lsampr, use_fp16=False):
  lpath_rip = get_lpath_rip_from_lpath_lfp(lpath_lfp_2kHz)
  rip_sec_df = pkl_load(lpath_rip) # Pandas.DataFrame

  if use_fp16:
    dtype = np.float16
  else:
    dtype = np.float32
  # lpath_lfp = replace_lpath_lfp_wrt_lsampr(lpath_lfp, lsampr)
  lpath_lfp = replace_lpath_lfp_wrt_lsampr_and_dtype(lpath_lfp_2kHz, lsampr, use_fp16)
  # lfp = np.load(lpath_lfp).squeeze().astype(dtype) # 2kHz -> int16, 1kHz, 500Hz -> float32
  lfp = np.load(lpath_lfp).squeeze() # 2kHz -> int16, 1kHz, 500Hz -> float32
  assert lfp.dtype == dtype
  return lfp, rip_sec_df, lsampr

def load_lfps_rips_sec_sampr(fpaths_2kHz, lsampr=1000, use_shuffle=False, use_fp16=False):
  # input: npy, pkl (pands.DataFrame) -> output: numpy
  lfps = []
  rips_sec = []
  for i in range(len(fpaths_2kHz)):
      lpath_lfp_2kHz = fpaths_2kHz[i]
      _2kHz_sampr = get_sampr_from_str(lpath_lfp_2kHz)
      assert _2kHz_sampr == 2000

      # lpath_lfp = replace_lpath_lfp_wrt_lsampr_and_dtype(lpath_lfp_2kHz, lsampr, use_fp16)
      lfp, rip_sec_df, sampr = load_lfp_rip_sec_sampr_from_lpath_lfp(lpath_lfp_2kHz, lsampr, use_fp16=use_fp16)

      lfps.append(lfp)
      rips_sec.append(rip_sec_df)

  if use_shuffle:
    lfps, rips_sec = shuffle(lfps, rips_sec) # 1st shuffle

  return lfps, rips_sec, lsampr

def define_Xb_Tb_from_lfp_and_rip_sec(lfp, rip_sec, sampr, max_seq_len, \
                                      step=None, use_perturb=False, use_fp16=False,
                                      use_shuffle=False):
  if use_fp16:
    dtype = np.float16
  else:
    dtype = np.float32

  rip_sec_cp = rip_sec.copy()

  if use_perturb:
    perturb = np.random.randint(0, max_seq_len) # < max_seq_len
  else:
    perturb = 0

  last_rip_start = int(rip_sec_cp.iloc[-1]['start_time'] * sampr)
  lfp = lfp[perturb:last_rip_start]

  if step == None:
    step = max_seq_len

  slices = util.view_as_windows(lfp, window_shape=(max_seq_len,), step=step)
  Xb = np.array(slices).astype(dtype)
  slices_end = np.array([perturb + max_seq_len + step*i for i in range(len(slices))])
  slices_end_sec = slices_end / sampr

  next_rip_idx = [bisect_left(rip_sec_cp['start_time'].values, slices_end_sec[i]) for i in range(len(slices_end_sec))]

  Tb_dur = np.array(rip_sec.iloc[next_rip_idx]['duration']).astype(dtype)
  Tb_lat = (np.array(rip_sec.iloc[next_rip_idx]['start_time']) - slices_end_sec).astype(dtype)

  if use_shuffle:
    Xb, Tb_dur, Tb_lat = shuffle(Xb, Tb_dur, Tb_lat) # 2nd shuffle

  return Xb, Tb_dur, Tb_lat

def define_Xb_Tb_from_lfp_and_rip_sec_wrapper(arg):
  args, kwargs = arg
  return define_Xb_Tb_from_lfp_and_rip_sec(*args, **kwargs)

def multi_define_Xb_Tb_from_lfps_and_rips(arg_list):
    n_cpus = 20 # mp.cpu_count()
    # print('n_cpus: {}'.format(n_cpus))
    p = mp.Pool(n_cpus)
    output = p.map(define_Xb_Tb_from_lfp_and_rip_sec_wrapper, arg_list)
    p.close()
    return output

# @Delogger.line_memory_profiler # @profile
def multi_define_Xb_Tb_from_lfps_and_rips_wrapper(lfps, rips_sec, sampr, max_seq_len, \
                                                  step=None, use_perturb=False, use_fp16=False,
                                                  use_shuffle=False):

  kwargs = {'step':step, 'use_perturb':use_perturb, 'use_fp16':use_fp16, \
            'use_shuffle':use_shuffle}
  arg_list = [((lfps[i], rips_sec[i], sampr, max_seq_len), kwargs) for i in range(len(lfps))]

  Xbs_and_Tbs = multi_define_Xb_Tb_from_lfps_and_rips(arg_list) # too big

  del arg_list, lfps, rips_sec
  gc.collect()

  Xb = []
  Tb_dur = []
  Tb_lat = []
  for i in range(len(Xbs_and_Tbs)):
    Xb_tmp, Tb_dur_tmp, Tb_lat_tmp = Xbs_and_Tbs[i]
    Xb.append(Xb_tmp)
    Tb_dur.append(Tb_dur_tmp)
    Tb_lat.append(Tb_lat_tmp)

  del Xbs_and_Tbs, Xb_tmp, Tb_dur_tmp, Tb_lat_tmp
  gc.collect()

  Xb = np.vstack(Xb)
  Tb_dur = np.hstack(Tb_dur)
  Tb_lat = np.hstack(Tb_lat)

  if use_shuffle: # 3rd shuffle
    Xb, Tb_dur, Tb_lat = shuffle(Xb, Tb_dur, Tb_lat)

  return Xb, Tb_dur, Tb_lat


def pack_to_dataloader(X, T1, T2, istrain=True, bs=64, use_fp16=False, nw=10, pm=True, collate_fn_class=None):
  X = torch.tensor(X).unsqueeze(-1)
  T1 = torch.tensor(T1)
  T2 = torch.tensor(T2)

  dataset = utils.TensorDataset(X, T1, T2) # X and T must be tensor

  if collate_fn_class:
    pass
    # # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/2
    # collate_fn = collate_fn_class(max_seq_len, sampr)
    # dataloader = utils.DataLoader(dataset, batch_size=bs, shuffle=train, num_workers=nw, \
    #                               pin_memory=True, collate_fn=collate_fn, drop_last=True)
  else:
    dataloader = utils.DataLoader(dataset, batch_size=bs, shuffle=istrain, num_workers=nw, \
                                  pin_memory=True, drop_last=True)
  return dataloader


def mk_dataloader(fpaths_2kHz, sampr, max_seq_len, step=None, use_perturb=False, bs=64, use_fp16=False, istrain=False, \
                  use_shuffle=False):
  lfps, rips_sec, sampr = load_lfps_rips_sec_sampr(fpaths_2kHz, lsampr=sampr, use_fp16=use_fp16)

  Xb, Tb_dur, Tb_lat = multi_define_Xb_Tb_from_lfps_and_rips_wrapper(lfps, rips_sec, sampr, max_seq_len, \
                                                                                      step=max_seq_len, \
                                                                                      use_perturb=use_perturb, \
                                                                                      use_fp16=use_fp16,\
                                                                                      use_shuffle=use_shuffle)

  dataloader = pack_to_dataloader(Xb, Tb_dur, Tb_lat,
                                  istrain=istrain, bs=bs, use_fp16=use_fp16, collate_fn_class=None, \
                                  nw=10, pm=True)
  del Xb, Tb_dur, Tb_lat
  gc.collect()
  return dataloader

class dataloader_fulfiller():
  def __init__(self, fpaths_2kHz, sampr, max_seq_len, step=None, use_perturb=False,
                     bs=64, use_fp16=False, istrain=False, use_shuffle=False):
    self.lfps, self.rips_sec, self.sampr = \
      load_lfps_rips_sec_sampr(fpaths_2kHz, lsampr=sampr, use_fp16=use_fp16)
    self.max_seq_len = max_seq_len
    if step == None:
      self.step = max_seq_len
    else:
      self.step = step
    self.use_perturb = use_perturb
    self.use_fp16 = use_fp16
    self.use_shuffle = use_shuffle
    self.istrain = istrain
    self.bs = bs

  def fulfill(self,):
    Xb, Tb_dur, Tb_lat = \
      multi_define_Xb_Tb_from_lfps_and_rips_wrapper(self.lfps, self.rips_sec, self.sampr,\
                                                                       self.max_seq_len,\
                                                                       step=self.step,\
                                                                       use_perturb=self.use_perturb, \
                                                                       use_fp16=self.use_fp16,\
                                                                       use_shuffle=self.use_shuffle)
    dataloader = pack_to_dataloader(Xb, Tb_dur, Tb_lat, istrain=self.istrain, bs=self.bs, use_fp16=self.use_fp16,
                                    collate_fn_class=None, nw=10, pm=True)
    return dataloader

  def get_n_samples(self,):
    dl = self.fulfill()
    return len(dl.dataset)


# sampr=1000
# max_seq_len=1000
# use_perturb=False
# bs=64
# use_fp16=False
# istrain=True
# isn_range = 100
# isn_under=True
# isn_over=False
# use_shuffle=True
# dl = mk_dataloader(p['fpaths_tra'], sampr, max_seq_len, isn_range, step=max_seq_len, use_perturb=use_perturb, bs=bs, \
#                    use_fp16=use_fp16, istrain=istrain, use_shuffle=True)
# dl_iterator = iter(dl)
# batch = next(dl_iterator)
# x, t1, t2, t3, t4 = batch
