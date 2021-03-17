import sys
sys.path.append('./')
import myutils.myfunc as mf
sys.path.append('./06_File_IO')
sys.path.append('./07_Learning/')
import os

from bisect import bisect_left, bisect_right
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
If you change ripple labels to feed, you have to change both define_Xb_Tb and collate_fn.
'''

# p = mf.listed_dict()
# loadpath_npy_list = '../data/1kHz_npy_list.pkl' # '../data/2kHz_npy_list.pkl'
# p['n_load_all'] = 12 # args.n_load_all # 186   # 12 -> fast, 176 -> w/o 05/day5, 186 -> full
# print('n_load_all : {}'.format(p['n_load_all']))
# fpaths = mf.pkl_load(loadpath_npy_list)[:p['n_load_all']]
# p['tes_keyword'] = '02' # args.n_mouse_tes # '02'
# print('Test Keyword: {}'.format(p['tes_keyword']))
# p['fpaths_tra'], p['fpaths_tes'] = mf.split_fpaths(fpaths, tes_keyword=p['tes_keyword'])
# print()
# pprint(p['fpaths_tra'])
# print()
# pprint(p['fpaths_tes'])
# print()

# keys_to_pack = ['Xb', 'Tb_CenterX_W', 'Tb_cls']
# kwargs = {'samp_rate':1000,
#           'use_fp16':True,
#           'use_shuffle':True,
#           'max_seq_len_pts':416,
#           'step':None,
#           'use_perturb':True,
#           'define_ripples':True,
#           'keys_to_pack':keys_to_pack,
#           'bs':64,
#           'nw':20,
#           'pm':True,
#           'drop_last':True,
#           }

# lfp_fpaths = p['fpaths_tra']


def parse_samp_rate(text):
  '''
  parse_samp_rate(p['fpaths_tra'][0]) # 1000
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
  kwargs = {'samp_rate':1000}
  test = cvt_lpath_lfp_2_lpath_rip(p['fpaths_tra'][0], **kwargs)
  '''
  samp_rate = parse_samp_rate(lpath_lfp)
  lsamp_str = cvt_samp_rate_int2str(**kwargs)
  lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').replace('_fp16.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm_filled.pkl')
  return lpath_rip


def load_lfp_rip_sec(lpath_lfp, **kwargs):
  '''
  kwargs = {'samp_rate':1000, 'use_fp16':True}
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
  kwargs = {'samp_rate':1000, 'use_fp16':True, 'use_shuffle':True}
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
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


def pad_sequence(listed_1Darrays, padding_value=0):
    '''
    listed_1Darrays = rips_level_in_slices
    '''
    listed_1Darrays = listed_1Darrays.copy()
    dtype = listed_1Darrays[0].dtype
    # get max_len
    max_len = 0
    for i in range(len(listed_1Darrays)):
      max_len = max(max_len, len(listed_1Darrays[i]))
    # padding
    for i in range(len(listed_1Darrays)):
      pad = (np.ones(max_len - len(listed_1Darrays[i])) * padding_value).astype(dtype)
      listed_1Darrays[i] = np.concatenate([listed_1Darrays[i], pad])
    listed_1Darrays = np.array(listed_1Darrays)
    return listed_1Darrays



def define_Xb_Tb(lfp, rip_sec, **kwargs):
  '''
  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':416,
            'step':None,
            'use_perturb':False,
            }
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tes'], **kwargs)
  lfp, rip_sec = lfps[0], rips_sec[0]

  outputs = define_Xb_Tb(lfp, rip_sec, **kwargs)
  print(outputs.keys())
  '''
  samp_rate = kwargs.get('samp_rate', 1000)
  max_seq_len_pts = kwargs.get('max_seq_len_pts', 416)

  dtype = np.float16 if kwargs.get('use_fp16', True) else np.float32

  perturb_pts = np.random.randint(0, max_seq_len_pts) if kwargs.get('use_perturb', False) else 0
  perturb_sec = perturb_pts / samp_rate


  cls_id = {'padding':-10,
            'back_ground':-2,
            'true_ripple':0,
            'false_ripple':1,
             }

  rip_sec_cp = rip_sec.copy()[['start_sec', 'end_sec', 'label_cleaned_from_gmm', 'prob_pred_by_ResNet']]\
                      .fillna(value={'label_cleaned_from_gmm':cls_id['back_ground']})
  rip_sec_cp_arr = np.array(rip_sec_cp)

  last_rip_start_sec = rip_sec_cp['start_sec'].iloc[-1]
  last_rip_start_pts = int(last_rip_start_sec * samp_rate)
  lfp = lfp[perturb_pts:last_rip_start_pts]

  step = kwargs.get('step') if kwargs.get('step') else max_seq_len_pts

  # Xb
  slices = util.view_as_windows(lfp, window_shape=(max_seq_len_pts,), step=step)

  slices_start_pts = np.array([perturb_pts + step*i for i in range(len(slices))]) + 1e-10
  slices_start_sec = slices_start_pts / samp_rate

  slices_center_pts = slices_start_pts + int(max_seq_len_pts/2)
  slices_center_sec = slices_center_pts / samp_rate

  slices_end_pts = slices_start_pts + max_seq_len_pts
  slices_end_sec = slices_end_pts / samp_rate

  on_the_start_rips_indi = np.array([bisect_left(rip_sec_cp['start_sec'].values, slices_start_sec[i]) -1 \
                                     for i in range(len(slices))])
  on_the_end_rips_indi = np.array([bisect_right(rip_sec_cp['end_sec'].values, slices_end_sec[i]) \
                                   for i in range(len(slices))])

  in_slice_rip_indi = [np.arange(on_the_start_rips_indi[i], on_the_end_rips_indi[i]+1) for i in range(len(slices))]


  rips_start_in_slices = [rip_sec_cp_arr[:,0][in_slice_rip_indi[i]] - slices_start_sec[i] for i in range(len(slices))]
  rips_end_in_slices = [rip_sec_cp_arr[:,1][in_slice_rip_indi[i]] - slices_start_sec[i] for i in range(len(slices))]

  rips_start_end_in_slices = [np.clip(
                              np.array([
                               rip_sec_cp_arr[:,0][in_slice_rip_indi[i]] - slices_start_sec[i],
                               rip_sec_cp_arr[:,1][in_slice_rip_indi[i]] - slices_start_sec[i]
                              ], dtype=dtype).T,
                              0, max_seq_len_pts / samp_rate)
                              for i in range(len(slices))]

  rips_centerx_in_slices = [np.array(rips_start_end_in_slices[i].mean(axis=1),
                            dtype=dtype) / (max_seq_len_pts/samp_rate)
                            for i in range(len(slices))]

  rips_w_in_slices = [np.array(rips_start_end_in_slices[i][:, 1] - rips_start_end_in_slices[i][:, 0],
                      dtype=dtype) / (max_seq_len_pts/samp_rate)
                      for i in range(len(slices))]


  rips_cls_in_slices = [np.array(rip_sec_cp_arr[:,2][in_slice_rip_indi[i]], dtype=np.int) \
                        for i in range(len(slices))]
  # rips_cls_in_slices = [np.array(rip_sec_cp_arr[:,2][in_slice_rip_indi[i]], dtype=np.float) \
  #                       for i in range(len(slices))]

  rips_label_conf_in_slices = [np.array(rip_sec_cp_arr[:,3][in_slice_rip_indi[i]], dtype=np.float) \
                             for i in range(len(slices))]

  ## Formatting
  Xb = np.array(slices)
  _Tb_CenterX = pad_sequence(rips_centerx_in_slices, padding_value=np.nan)
  _Tb_W = pad_sequence(rips_w_in_slices, padding_value=np.nan)
  Tb_CenterX_W = np.concatenate([_Tb_CenterX[..., np.newaxis], _Tb_W[..., np.newaxis]], axis=-1)
  Tb_cls = pad_sequence(rips_cls_in_slices, padding_value=cls_id['padding'])
  Tb_label_conf = pad_sequence(rips_label_conf_in_slices, padding_value=np.nan)

  '''
  Class List as described in cls_id
  -10: padding,
   -2: Back ground,
    0: "True" Ripples
    1: High-EMG False Ripple candi.,
  '''

  ########## Select training data ##########

  ## Select the slices including either "True" Ripples or High-EMG False Ripple candidates
  include_true_ripples = (Tb_cls == cls_id['true_ripple']).any(axis=1)
  include_false_ripples = (Tb_cls == cls_id['false_ripple']).any(axis=1)
  indi = (include_true_ripples + include_false_ripples)
  Xb, Tb_CenterX_W, Tb_cls, Tb_label_conf = Xb[indi], Tb_CenterX_W[indi], Tb_cls[indi], Tb_label_conf[indi]

  ## Filter out Width == 0 samples
  indi = (Tb_CenterX_W[:,:,1] != 0).any(axis=1)
  Xb, Tb_CenterX_W, Tb_cls, Tb_label_conf = Xb[indi], Tb_CenterX_W[indi], Tb_cls[indi], Tb_label_conf[indi]

  # Select samples which start and end at not ripple-candidate time
  indi_bg_start = (Tb_cls[:, 0] == cls_id['back_ground'])
  pad_len = (Tb_cls == cls_id['padding']).sum(axis=-1)
  last_cls = np.array([Tb_cls[i, Tb_cls.shape[-1] - (pad_len[i]+1)] for i in range(len(pad_len))])
  indi_bg_end = (last_cls == cls_id['back_ground'])
  indi = indi_bg_start & indi_bg_end
  Xb, Tb_CenterX_W, Tb_cls, Tb_label_conf = Xb[indi], Tb_CenterX_W[indi], Tb_cls[indi], Tb_label_conf[indi]

  # Mask out samples which won't be used
  not_used_mask = (Tb_cls != cls_id['true_ripple']) & (Tb_cls != cls_id['false_ripple'])
  Tb_cls[not_used_mask], Tb_CenterX_W[not_used_mask], Tb_label_conf[not_used_mask] = -1, np.nan, np.nan

  # Sorting
  indi_sort = np.argsort(Tb_cls)[...,::-1]
  Tb_cls = np.take_along_axis(Tb_cls, indi_sort, axis=-1)
  Tb_label_conf = np.take_along_axis(Tb_label_conf, indi_sort, axis=-1)
  Tb_CenterX_W = np.take_along_axis(Tb_CenterX_W, indi_sort[..., np.newaxis], axis=1)

  outputs = {'Xb':Xb, 'Tb_CenterX_W':Tb_CenterX_W, 'Tb_cls':Tb_cls, 'Tb_label_conf':Tb_label_conf}

  if kwargs['use_shuffle']: # 2nd shuffle
    outputs = mf.shuffle_dict(outputs)

  return outputs


def define_Xb_Tb_wrapper(arg_list):
  args, kwargs = arg_list
  return define_Xb_Tb(*args, **kwargs)


def multi_define_Xb_Tb(arg_list):
    '''
    kwargs = {'samp_rate':1000,
              'use_fp16':True,
              'use_shuffle':True,
              'max_seq_len_pts':416,
              'step':None,
              'use_perturb':True,
              'define_ripples':True,
              }
    lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
    arg_list = [((lfps[i], rips_sec[i]), kwargs) for i in range(len(lfps))] # args, kwargs
    Xb_Tbs_dict_list = multi_define_Xb_Tb(arg_list)
    print(Xb_Tbs_dict_list[0]['Xb'])
    '''
    n_cpus = 20 # mp.cpu_count()
    p = mp.Pool(n_cpus)
    # Multiprocessing passes objects between processes by pickling them.
    # Maybe you have stuff in your function that isn't safe to paralelize and it's breaking your program.
    output = p.map(define_Xb_Tb_wrapper, arg_list) # IndexError: index 121426 is out of bounds for axis 0 with size 121426
    # imap, map_async, imap_unordered
    p.close()
    p.join()
    return output


# @Delogger.line_memory_profiler # @profile
def multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs):
  '''
  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':416,
            'step':None,
            'use_perturb':True,
            'define_ripples':True,
            }
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)

  Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs)
  print(Xb_Tbs_dict['Tb_cls'])
  print(Xb_Tbs_dict['Tb_label_conf'])
  '''

  arg_list = [((lfps[i], rips_sec[i]), kwargs) for i in range(len(lfps))] # args, kwargs
  Xb_Tbs_dict_list = multi_define_Xb_Tb(arg_list) # too big

  del arg_list, lfps, rips_sec
  gc.collect()

  def gathers_mp_outputs_dict_list(mp_outputs_dict_list, **kwargs): # Xb_Tb_keys=None
    mp_output_keys = mp_outputs_dict_list[0].keys()
    n_keys = len(mp_output_keys)
    assert n_keys == len(mp_outputs_dict_list[0])

    gathered_dict = mf.listed_dict(mp_output_keys)
    for i in range(len(mp_outputs_dict_list)):
      for k in mp_output_keys:
        gathered_dict[k].append(mp_outputs_dict_list[i][k])

    # Xb
    gathered_dict['Xb'] = np.concatenate(gathered_dict['Xb'])

    max_n_bounding_ranges_in_an_input = max(np.array([gathered_dict['Tb_CenterX_W'][i].shape[1]
                                                      for i in range(len(gathered_dict['Tb_CenterX_W']))]))

    # Tb_CenterX_W
    dtype = gathered_dict['Tb_CenterX_W'][0].dtype
    for i in range(len(gathered_dict['Tb_CenterX_W'])):
      s1, s2, s3 = gathered_dict['Tb_CenterX_W'][i].shape
      if max_n_bounding_ranges_in_an_input - s2 > 0:
        nans_arr = (np.ones([s1, max_n_bounding_ranges_in_an_input - s2, s3]) * np.nan).astype(dtype)
        gathered_dict['Tb_CenterX_W'][i] = np.concatenate([gathered_dict['Tb_CenterX_W'][i], nans_arr], axis=1)
    gathered_dict['Tb_CenterX_W'] = np.concatenate(gathered_dict['Tb_CenterX_W'])

    # Tb_cls
    dtype = gathered_dict['Tb_cls'][0].dtype
    for i in range(len(gathered_dict['Tb_cls'])):
      s1, s2 = gathered_dict['Tb_cls'][i].shape
      if max_n_bounding_ranges_in_an_input - s2 > 0:
        minus_ones_arr = (np.ones([s1, max_n_bounding_ranges_in_an_input - s2]) * -1).astype(dtype)
        gathered_dict['Tb_cls'][i] = np.concatenate([gathered_dict['Tb_cls'][i], minus_ones_arr], axis=1)
    gathered_dict['Tb_cls'] = np.concatenate(gathered_dict['Tb_cls'])

    # Tb_label_conf
    dtype = gathered_dict['Tb_label_conf'][0].dtype
    for i in range(len(gathered_dict['Tb_label_conf'])):
      s1, s2 = gathered_dict['Tb_label_conf'][i].shape
      if max_n_bounding_ranges_in_an_input - s2 > 0:
        nans_arr = (np.ones([s1, max_n_bounding_ranges_in_an_input - s2]) * np.nan).astype(dtype)
        gathered_dict['Tb_label_conf'][i] = np.concatenate([gathered_dict['Tb_label_conf'][i], nans_arr], axis=1)
    gathered_dict['Tb_label_conf'] = np.concatenate(gathered_dict['Tb_label_conf'])

    first = True
    for v in gathered_dict.values():
      if first:
        length = len(v)
        first = False
      else:
        assert length == len(v)

    return gathered_dict

  Xb_Tbs_dict = gathers_mp_outputs_dict_list(Xb_Tbs_dict_list, **kwargs)

  if kwargs['use_shuffle']: # 3rd shuffle
    Xb_Tbs_dict = mf.shuffle_dict(Xb_Tbs_dict)

  return Xb_Tbs_dict



def _check_samples(Xb, targets, label_conf=None, max_plot=1):
    plot = 0
    for _ in range(1000):
      ix = np.random.randint(len(Xb))
      x = Xb[ix].squeeze()
      indi_targets = (targets[:,0] == ix)
      _targets = targets[indi_targets]
      classes = _targets[:,1]
      if type(label_conf) != type(None):
          _label_conf = label_conf[indi_targets]
          title = 'Classes: {} \n Confidence: {:.2f}'.format(classes.long().item(), _label_conf.squeeze().item())
      else:
          title = 'Classes: {}'.format(classes.long().item())
      fig, ax = plt.subplots()
      ax.plot(np.arange(len(x)), x)

      for i_target in range(len(_targets)):
        obj = _targets[i_target]
        cls, X, W = obj[1], obj[2], obj[3]
        X, W = X*len(x), W*len(x)
        left = int(X - W/2)
        right = int(X + W/2)
        ax.axvspan(left, right, alpha=0.3, color='red', zorder=1000)

      plt.title(title)

      plot += 1
      if plot == max_plot:
        break


def pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs):
  '''
  keys_to_pack = ['Xb', 'Tb_CenterX_W', 'Tb_cls', 'Tb_label_conf']
  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':52*8*4,
            'step':None,
            'use_perturb':True,
            'define_ripples':True,
            'keys_to_pack':keys_to_pack,
            'bs':64,
            'nw':10,
            'pm':True,
            'drop_last':True,
            }

  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
  Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs)

  dataloader = pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs)
  batch = next(iter(dataloader))
  Xb, targets_with_label_conf = batch
  _check_samples(Xb, targets, max_plot=30)
  '''
  # dataset = utils.TensorDataset(*(Xb_Tbs_dict[k] for k in kwargs['keys_to_pack']))
  dataset = utils.TensorDataset(*(torch.tensor(Xb_Tbs_dict[k]) for k in kwargs['keys_to_pack']))

  def collate_fn(batch):
      '''
      Transform Tb_CenterX_W and Tb_cls for YOLO.
      '''
      Xb = torch.stack([item[0] for item in batch], dim=0).unsqueeze(-1)
      Tb_CenterX_W = torch.stack([item[1] for item in batch], dim=0)
      Tb_cls = torch.stack([item[2] for item in batch], dim=0)
      Tb_label_conf = torch.stack([item[3] for item in batch], dim=0)

      targets_mask = (Tb_cls != -1) # -1 means back ground
      ids = torch.arange(len(targets_mask)).unsqueeze(-1).expand(-1, Tb_CenterX_W.shape[1])
      dtype = Xb.dtype

      targets_ids = ids[targets_mask].unsqueeze(-1)
      targets_cls = Tb_cls[targets_mask].unsqueeze(-1)
      targets_label_conf = Tb_label_conf[targets_mask].unsqueeze(-1)
      targets_X_W = Tb_CenterX_W[targets_mask, :]

      # targets = torch.cat([targets_ids.to(dtype), targets_cls.to(dtype), targets_X_W], dim=-1) # YOLO's format
      targets_with_label_conf = torch.cat([targets_ids.to(dtype),
                                           targets_cls.to(dtype),
                                           targets_X_W.to(dtype),
                                           targets_label_conf.to(dtype),
                                          ], dim=-1) # YOLO's format + label_conf

      # To fp32
      Xb, targets_with_label_conf = Xb.transpose(1,2).float(), targets_with_label_conf.float()

      return Xb, targets_with_label_conf

  dataloader = utils.DataLoader(dataset,
                                batch_size=kwargs['bs'],
                                shuffle=kwargs['use_shuffle'],
                                num_workers=kwargs['nw'],
                                pin_memory=kwargs['pm'],
                                drop_last=kwargs['drop_last'],
                                collate_fn=collate_fn)
  return dataloader


def mk_dataloader(lfp_fpaths, **kwargs):
  '''
  keys_to_pack = ['Xb', 'Tb_CenterX_W', 'Tb_cls', 'Tb_label_conf']
  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':416,
            'step':None,
            'use_perturb':True,
            'define_ripples':True,
            'keys_to_pack':keys_to_pack,
            'bs':64,
            'nw':10,
            'pm':True,
            'drop_last':True,
            }

  dataloader = mk_dataloader(p['fpaths_tra'], **kwargs)# pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs)
  batch = next(iter(dataloader))
  Xb, targets = batch
  _check_samples(Xb, targets, label_conf=label_conf)
  '''
  lfps, rips_sec = load_lfps_rips_sec(lfp_fpaths, **kwargs)
  Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs)
  dataloader = pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs)
  del Xb_Tbs_dict
  gc.collect()
  return dataloader


class DataloaderFulfiller():
  '''
  keys_to_pack = ['Xb', 'Tb_CenterX_W', 'Tb_cls', 'Tb_label_conf']
  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':416,
            'step':None,
            'use_perturb':True,
            'define_ripples':True,
            'keys_to_pack':keys_to_pack,
            'bs':64,
            'nw':20,
            'pm':True,
            'drop_last':True,
            }

  lfp_fpaths = p['fpaths_tra']
  dl_fulf = DataloaderFulfiller(lfp_fpaths, **kwargs)

  n_samples = dl_fulf.get_n_samples()
  print(n_samples)

  dl = dl_fulf()
  batch = next(iter(dl))
  Xb, targets = batch
  _check_samples(Xb, targets[:,:-1], label_conf=targets[:,-1], max_plot=10)
  '''
  def __init__(self, lfp_fpaths, **kwargs):
    self.lfps, self.rips_sec = load_lfps_rips_sec(lfp_fpaths, **kwargs)
    self.kwargs = kwargs
    self.n_samples = None
    self.dl = None
    self.just_has_fullfilled_by_n_sample_check = False

  def __call__(self,):
    if self.just_has_fullfilled_by_n_sample_check:
      self.just_has_fullfilled_by_n_sample_check = False
      return self.dl

    if not self.just_has_fullfilled_by_n_sample_check:
      self.dl = None
      Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(self.lfps, self.rips_sec, **self.kwargs)
      dl = pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **self.kwargs)
      return dl

  def get_n_samples(self,): # fixme
    if self.n_samples is not None:
      return self.n_samples

    if self.n_samples is None:

      if self.dl is None:
        # self.dl = self.fulfill()
        self.dl = self()
        self.n_samples = len(self.dl.dataset)
        self.just_has_fullfilled_by_n_sample_check = True
        return self.n_samples

      if self.dl is not None:
        self.n_samples = len(self.dl.dataset)
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

# def cvt_Tb_distance_ms(Tb_distances_ms, max_distance_ms):
#   Tb_distances_ms[Tb_distances_ms == -np.inf] = -(max_distance_ms+1)
#   Tb_distances_ms[Tb_distances_ms == np.inf] = (max_distance_ms+1)
#   Tb_distances_ms += max_distance_ms+1

#   Tb_distances_onehot = one_hot_embedding(Tb_distances_ms.to(torch.long), (max_distance_ms+1)*2+1)
#   return Tb_distances_onehot


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










if __name__ == '__main__':
  ## Load Paths
  p = mf.listed_dict()
  loadpath_npy_list = '../data/1kHz_npy_list.pkl' # '../data/2kHz_npy_list.pkl'
  p['n_load_all'] = 186 #12 # args.n_load_all # 186   # 12 -> fast, 176 -> w/o 05/day5, 186 -> full
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

  keys_to_pack = ['Xb', 'Tb_CenterX_W', 'Tb_cls']
  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':416,
            'step':None,
            'use_perturb':True,
            'define_ripples':True,
            'keys_to_pack':keys_to_pack,
            'bs':64,
            'nw':20,
            'pm':True,
            'drop_last':True,
            }

  lfp_fpaths = p['fpaths_tra']
  ####################################################
  '''
  dl_fulf = DataloaderFulfiller(lfp_fpaths, **kwargs)

  n_samples = dl_fulf.get_n_samples()
  print(n_samples)

  dl = dl_fulf()
  batch = next(iter(dl))
  Xb, targets = batch
  _check_samples(Xb, targets, max_plot=100)
  '''
