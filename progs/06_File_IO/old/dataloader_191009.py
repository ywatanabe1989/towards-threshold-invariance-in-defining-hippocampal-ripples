import sys
sys.path.append('./')
import utils.myfunc as mf
sys.path.append('./06_File_IO')
sys.path.append('./07_Learning/')
import os

import gc
import multiprocessing as mp
import numpy as np
import torch
import torch.utils.data as utils
from pprint import pprint

from sklearn.utils import shuffle
from skimage import util

from tqdm import tqdm
import math

import socket
hostname = socket.gethostname()
if hostname == 'localhost.localdomain':
  ###
  from delogger import Delogger
  Delogger.is_debug_stream = True
  debuglog = Delogger.line_profiler
  ###

'''
##################################################
## Load Paths
p = mf.listed_dict()
loadpath_npy_list = '../data/1kHz_npy_list.pkl' # '../data/2kHz_npy_list.pkl'
p['n_load_all'] = 12 # args.n_load_all # 186   # 12 -> fast, 176 -> w/o 05/day5, 186 -> full
print('n_load_all : {}'.format(p['n_load_all']))
fpaths = mf.pkl_load(loadpath_npy_list)[:p['n_load_all']]
p['tes_keyword'] = '02' # args.n_mouse_tes # '02'
print('Test Keyword: {}'.format(p['tes_keyword']))
p['fpaths_tra'], p['fpaths_tes'] = mf.split_fpaths(fpaths, tes_keyword=p['tes_keyword'])
print()
pprint(p['fpaths_tra'])
print()
pprint(p['fpaths_tes'])
print()
##################################################
'''


def glob_samp_rate(text):
  '''
  glob_samp_rate(p['fpaths_tra'][0]) # 1000
  '''
  if text.find('2kHz') >= 0:
    samp_rate = 2000
  if text.find('1kHz') >= 0:
    samp_rate = 1000
  if text.find('500Hz') >= 0:
    samp_rate = 500
  return samp_rate


def cvt_samp_rate_int2str(**kwargs):
  '''
  kwargs = {'samp_rate':500}
  cvt_samp_rate_int2str(**kwargs) # '500Hz'
  '''
  samp_rate = kwargs.get('samp_rate', 1000)
  samp_rate_estr = '{:e}'.format(samp_rate)
  e = int(samp_rate_estr[-1])
  if e == 3:
    add_str = 'kHz'
  if e == 2:
    add_str = '00Hz'
  samp_rate_str = samp_rate_estr[0] + add_str
  return samp_rate_str


def cvt_lpath_lfp_2_lpath_rip(lpath_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'sd':2}
  cvt_lpath_lfp_2_lpath_rip(p['fpaths_tra'][0], **kwargs)
  '''
  samp_rate = glob_samp_rate(lpath_lfp)
  lsamp_str = cvt_samp_rate_int2str(**kwargs)
  lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').replace('.npy', '_riptimes_sd{}.pkl'.format(kwargs.get('sd', 5)))
  return lpath_rip


def load_lfp_rip_sec(lpath_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'sd':2, 'use_fp16':True}
  lpath_lfp = p['fpaths_tra'][0]
  lfp, rip_sec = load_lfp_rip_sec(p['fpaths_tra'][0], **kwargs)
  '''
  use_fp16 = kwargs.get('use_fp16', True)

  dtype = np.float16 if use_fp16 else np.float32

  lpath_lfp = lpath_lfp.replace('.npy', '_fp16.npy') if use_fp16 else lpath_lfp
  lpath_rip = cvt_lpath_lfp_2_lpath_rip(lpath_lfp, **kwargs)

  lfp = np.load(lpath_lfp).squeeze().astype(dtype) # 2kHz -> int16, 1kHz, 500Hz -> float32
  rip_sec_df = mf.pkl_load(lpath_rip).astype(float) # Pandas.DataFrame

  return lfp, rip_sec_df


def load_lfps_rips_sec(lpaths_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'sd':2, 'use_fp16':True, 'use_shuffle':True}
  load_lfps_rips_sec(p['fpaths_tes'], **kwargs)
  '''
  lfps = []
  rips_sec = []
  for i in range(len(lpaths_lfp)):
      lpath_lfp = lpaths_lfp[i]
      lfp, rip_sec_df = load_lfp_rip_sec(lpath_lfp, **kwargs)
      lfps.append(lfp)
      rips_sec.append(rip_sec_df)

  if kwargs.get('use_shuffle', False):
    lfps, rips_sec = shuffle(lfps, rips_sec) # 1st shuffle

  return lfps, rips_sec


def define_Xb_Tb(lfp, rip_sec, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'sd':2, 'use_fp16':True, 'use_shuffle':True,
            'max_seq_len_pts':200, 'step':None, 'use_perturb':True,
            'max_distance_ms':None,
            'detects_ripples':True, 'estimates_ripple_params':False,
            }
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tes'], **kwargs)
  lfp, rip_sec = lfps[0], rips_sec[0]
  outputs = define_Xb_Tb(lfp, rip_sec, **kwargs)
  print(outputs)
  '''

  def get_in_slice_distances_ms_ripple_idx(ripple_peak_posis_sec, slice_center_sec, max_distance_ms):
    _closest_ripple_peak_posi_sec, _closest_ripple_idx = mf.take_closest(ripple_peak_posis_sec, slice_center_sec)
    distance_ms = (_closest_ripple_peak_posi_sec - slice_center_sec) * kwargs.get('samp_rate', 1000)

    if max_distance_ms < distance_ms:
      return np.inf, np.nan

    if distance_ms < -max_distance_ms:
      return -np.inf, -np.nan

    else:
      in_slice_distance_ms = distance_ms
      in_slice_ripple_idx = _closest_ripple_idx
      return in_slice_distance_ms, in_slice_ripple_idx

  samp_rate = kwargs.get('samp_rate', 1000)
  max_seq_len_pts = kwargs.get('max_seq_len_pts', 200)
  max_distance_ms = kwargs.get('max_distance_ms') if kwargs.get('max_distance_ms') else int(max_seq_len_pts/2)
  dtype = np.float16 if kwargs.get('use_fp16', True) else np.float32

  perturb_pts = np.random.randint(0, max_seq_len_pts) if kwargs.get('use_perturb', False) else 0
  perturb_sec = perturb_pts / samp_rate

  rip_sec_cp = rip_sec.copy()
  rip_sec_cp = rip_sec_cp[perturb_sec < rip_sec_cp['start_sec']]

  last_rip_start_pts = int(rip_sec_cp.iloc[-1]['start_sec'] * samp_rate)
  lfp = lfp[perturb_pts:last_rip_start_pts]

  step = kwargs.get('step') if kwargs.get('step') else max_seq_len_pts

  # Xb
  slices = util.view_as_windows(lfp, window_shape=(max_seq_len_pts,), step=step)
  Xb = np.array(slices).astype(dtype)

  slices_starts_pts = np.array([perturb_pts + step*i for i in range(len(slices))])
  slices_starts_sec = slices_starts_pts / samp_rate

  slices_centers_pts = slices_starts_pts + int(max_seq_len_pts/2)
  slices_centers_sec = slices_centers_pts / samp_rate

  slices_ends_pts = slices_starts_pts + max_seq_len_pts
  slices_ends_sec = slices_ends_pts / samp_rate
  # next_rip_indi_list = [bisect_left(rip_sec_cp['start_sec'].values, slices_ends_sec[i]) for i in range(len(slices_ends_sec))]

  # from Xb to rip_sec
  _tmp_outputs = np.array([get_in_slice_distances_ms_ripple_idx(rip_sec_cp['ripple_peak_posi_sec'].values,
                                                                slices_centers_sec[i_slice],
                                                                max_distance_ms) \
                            for i_slice in range(len(slices_centers_sec))])
  in_slices_distances_ms, in_slices_ripple_indi = _tmp_outputs[:, 0], _tmp_outputs[:, 1]

  Tb_isRipple = ~np.isnan(in_slices_ripple_indi)

  Tb_distances_ms = in_slices_distances_ms

  assert len(Xb) == len(Tb_isRipple) == len(Tb_distances_ms)

  assert kwargs['detects_ripples'] != kwargs['estimates_ripple_params']

  if kwargs['detects_ripples']:
    # outputs = (Xb, Tb_distances_ms, Tb_isRipple)
    outputs = {'Xb':Xb, 'Tb_distances_ms':Tb_distances_ms, 'Tb_isRipple':Tb_isRipple}

  if kwargs['estimates_ripple_params']:
    '''
    In this case, Slices which don't include ripple peaks are useless, since parameter labels don't exist.
    So, Xb, Tb_distances_ms, in_slices_ripple_indi and slices_starts_sec must be "pruned" wrt Tb_isRipple.
    '''

    Xb = Xb[Tb_isRipple]
    Tb_distances_ms = Tb_distances_ms[Tb_isRipple]
    in_slices_ripple_indi = in_slices_ripple_indi[Tb_isRipple].astype(np.int)
    slices_starts_sec = slices_starts_sec[Tb_isRipple]

    # packing to numpy arrays
    keys = list(rip_sec.keys())
    dict_container = mf.listed_dict(keys)

    for i_slice in range(len(Xb)):
        i_rip = in_slices_ripple_indi[i_slice]
        for k in keys:
          if '_sec' in k:
            rip_sec_cp.iloc[i_rip][k] -= slices_starts_sec[i_slice]
          dict_container[k].append(rip_sec_cp.iloc[i_rip][k])

    for k in keys:
      dict_container[k] = np.array(dict_container[k]).astype(dtype)

    dict_container.update({'Xb':Xb, 'Tb_distances_ms':Tb_distances_ms})
    outputs = dict_container

  if kwargs['use_shuffle']: # 2nd shuffle
    outputs = mf.shuffle_dict(outputs)

  return outputs


def define_Xb_Tb_wrapper(arg):
  args, kwargs = arg
  return define_Xb_Tb(*args, **kwargs)

def multi_define_Xb_Tb(arg_list):
    n_cpus = 20 # mp.cpu_count()
    # print('n_cpus: {}'.format(n_cpus))
    p = mp.Pool(n_cpus)
    output = p.map(define_Xb_Tb_wrapper, arg_list)
    p.close()
    return output

# @Delogger.line_memory_profiler # @profile
def multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'sd':5, 'use_fp16':True, 'use_shuffle':True,
            'max_seq_len_pts':200, 'step':None, 'use_perturb':True,
            'max_distance_ms':None,
            'detects_ripples':False, 'estimates_ripple_params':True,
            }
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
  Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs)

  for _ in range(1000):
    i = np.random.randint(1e3)
    try:
      isripple = Xb_Tbs_dict['Tb_isRipple'][i]
    except:
      isripple = True
    if isripple:
      plt.plot(np.arange(-100, 100), Xb_Tbs_dict['Xb'][i])
      plt.title('Distance: {} ms'.format(Xb_Tbs_dict['Tb_distances_ms'][i]))
      break
  '''

  arg_list = [((lfps[i], rips_sec[i]), kwargs) for i in range(len(lfps))]

  Xb_Tbs_dict = multi_define_Xb_Tb(arg_list) # too big

  del arg_list, lfps, rips_sec
  gc.collect()

  def gathers_multiprocessing_output(output, **kwargs): # Xb_Tb_keys=None
    Xb_Tb_keys = output[0].keys()
    n_keys = len(Xb_Tb_keys)
    assert n_keys == len(output[0])

    dict_container = mf.listed_dict(Xb_Tb_keys)

    for i in range(len(output)):
      for k in Xb_Tb_keys:
        dict_container[k].append(output[i][k])

    for k in Xb_Tb_keys:
      if k == 'Xb':
        dict_container[k] = np.vstack(dict_container[k])
      else:
        dict_container[k] = np.hstack(dict_container[k])

    first = True
    for v in dict_container.values():
      if first:
        length = len(v)
        first = False
      else:
        assert length == len(v)

    return dict_container

  Xb_Tbs_dict = gathers_multiprocessing_output(Xb_Tbs_dict, **kwargs)

  if kwargs['use_shuffle']: # 3rd shuffle
    Xb_Tbs_dict = mf.shuffle_dict(Xb_Tbs_dict)

  return Xb_Tbs_dict


def pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs):
  '''
  keys_to_pack_detects_ripples = ['Xb', 'Tb_distances_ms', 'Tb_isRipple']
  keys_to_pack_estimates_ripple_params = ['Xb', 'Tb_distances_ms', 'ripple_relat_peak_posi', 'ripple_ave_power',
                                          'ripple_peak_power', 'ripple_peak_frequency_hz', 'gamma_ave_power']
  kwargs = {'samp_rate':1000, 'sd':5, 'use_fp16':True, 'use_shuffle':True,
            'max_seq_len_pts':200, 'step':None, 'use_perturb':True,
            'max_distance_ms':None,
            'detects_ripples':False, 'estimates_ripple_params':True,
            'bs':64, 'nw':10, 'pm':True, 'drop_last':True, 'collate_fn_class':None,
            'keys_to_pack_detects_ripples':keys_to_pack_detects_ripples,
            'keys_to_pack_estimates_ripple_params':keys_to_pack_estimates_ripple_params,
            }
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
  Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs)
  dataloader = pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs)
  '''
  keys_to_pack = kwargs['keys_to_pack_detects_ripples'] if kwargs['detects_ripples'] \
                                                        else kwargs['keys_to_pack_estimates_ripple_params']

  dataset = utils.TensorDataset(*(torch.tensor(Xb_Tbs_dict[k]) for k in keys_to_pack))

  if kwargs['collate_fn_class']:
    pass
    # # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/2
    # collate_fn = collate_fn_class(max_seq_len_pts, samp_rate)
    # dataloader = utils.DataLoader(dataset, batch_size=bs, shuffle=train, num_workers=nw, \
    #                               pin_memory=True, collate_fn=collate_fn, drop_last=True)
  else:
    dataloader = utils.DataLoader(dataset,
                                  batch_size=kwargs['bs'],
                                  shuffle=kwargs['use_shuffle'],
                                  num_workers=kwargs['nw'],
                                  pin_memory=kwargs['pm'],
                                  drop_last=kwargs['drop_last'])
  return dataloader


def mk_dataloader(lfp_fpaths, **kwargs):
  '''
  keys_to_pack_detects_ripples = ['Xb', 'Tb_distances_ms', 'Tb_isRipple']
  keys_to_pack_estimates_ripple_params = ['Xb', 'Tb_distances_ms', 'ripple_relat_peak_posi', 'ripple_ave_power',
                                          'ripple_peak_power', 'ripple_peak_frequency_hz', 'gamma_ave_power']
  kwargs = {'samp_rate':1000, 'sd':5, 'use_fp16':True, 'use_shuffle':True,
            'max_seq_len_pts':200, 'step':None, 'use_perturb':True,
            'max_distance_ms':None,
            'detects_ripples':True, 'estimates_ripple_params':False,
            'bs':64, 'nw':10, 'pm':True, 'drop_last':True, 'collate_fn_class':None,
            'keys_to_pack_detects_ripples':keys_to_pack_detects_ripples,
            'keys_to_pack_estimates_ripple_params':keys_to_pack_estimates_ripple_params,
            }
  dataloader = mk_dataloader(p['fpaths_tra'], **kwargs)
  next(iter(dataloader))
  '''

  lfps, rips_sec = load_lfps_rips_sec(lfp_fpaths, **kwargs)
  Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs)
  dataloader = pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs)
  del Xb_Tbs_dict
  gc.collect()
  return dataloader


class dataloader_fulfiller():
  '''
  keys_to_pack_detects_ripples = ['Xb', 'Tb_distances_ms', 'Tb_isRipple']
  keys_to_pack_estimates_ripple_params = ['Xb', 'Tb_distances_ms', 'ripple_relat_peak_posi', 'ripple_ave_power',
                                          'ripple_peak_power', 'ripple_peak_frequency_hz', 'gamma_ave_power']
  kwargs = {'samp_rate':1000, 'sd':5, 'use_fp16':True, 'use_shuffle':True,
            'max_seq_len_pts':200, 'step':None, 'use_perturb':True,
            'max_distance_ms':None,
            'detects_ripples':True, 'estimates_ripple_params':False,
            'bs':64, 'nw':10, 'pm':True, 'drop_last':True, 'collate_fn_class':None,
            'keys_to_pack_detects_ripples':keys_to_pack_detects_ripples,
            'keys_to_pack_estimates_ripple_params':keys_to_pack_estimates_ripple_params,
            }
  lfp_fpaths = p['fpaths_tra']
  dl_fulf = dataloader_fulfiller(lfp_fpaths, **kwargs)
  dl_fulf.get_n_samples()
  dl = dl_fulf.fulfill()
  batch = next(iter(dl))
  '''
  def __init__(self, lfp_fpaths, **kwargs):
    self.lfps, self.rips_sec = \
      load_lfps_rips_sec(lfp_fpaths, **kwargs)
    self.kwargs = kwargs
    self.n_samples = None

  def fulfill(self,):
    Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(self.lfps, self.rips_sec, **self.kwargs)
    dataloader = pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **self.kwargs)
    return dataloader

  def get_n_samples(self,):
    if self.n_samples == None:
      dl = self.fulfill()
      self.n_samples = len(dl.dataset)
    return self.n_samples




def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def to_one_hot(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = np.eye(num_classes)
    return y[labels]


def cvt_Tb_distance_ms(Tb_distances_ms, max_distance_ms):
  Tb_distances_ms[Tb_distances_ms == -np.inf] = -(max_distance_ms+1)
  Tb_distances_ms[Tb_distances_ms == np.inf] = (max_distance_ms+1)
  Tb_distances_ms += max_distance_ms+1

  Tb_distances_onehot = one_hot_embedding(Tb_distances_ms.to(torch.long), (max_distance_ms+1)*2+1)
  return Tb_distances_onehot


'''
import matplotlib.pyplot as plt
from PIL import Image
import cv2
test = lfp[:200]
def vec2im(vec):
  vec = test
  vec /= (vec.max() - vec.min())
  vec -= vec.min()

  w, h = 416, 416
  vec *= (h-1)
  vec = vec.astype(np.int)

  vec = to_one_hot(vec, w)
  # vec = vec.cpu().numpy()

  rows, cols = vec.shape[:2]

  src_points = np.float32([[0,0], [cols-1, 0], [0, rows-1]])
  dst_points = np.float32([[0,0], [h-1, 0], [0, w-1]])

  affine_matrix = cv2.getAffineTransform(src_points, dst_points)
  img_output = cv2.warpAffine(vec, affine_matrix, (cols,rows))
  return img_output

im = vec2im(test)
'''
