import sys
sys.path.append('./')
import myutils.myfunc as mf
sys.path.append('./05_Ripple/rippledetection')
from core import gaussian_smooth

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
import pandas as pd

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

# # keys_to_pack = ['Xb', 'Tb_CenterX_W', 'Tb_cls']
# keys_to_pack = ['Xb', 'Tb_hm', 'Tb_w']
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
#           'smoothing_sigma':0.004,
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
  lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').\
    replace('_fp16.npy', '_ripple_candi_150-250Hz_with_prop_label_cleaned_from_gmm_merged_filled.pkl')
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
    # '''
    # Here, we should only choose samples to pack to dataloaders for total training speed.
    # Formatting will be conducted by collate function, which works asynchronously with GPU calculation.
    # '''
    '''
    keys_to_pack = ['Xb', 'Tb_c', 'Tb_w']
    kwargs = {'samp_rate':1000,
              'use_fp16':True,
              'use_shuffle':True,
              'max_seq_len_pts':512,
              'step':None,
              'use_perturb':False,
              'n_mouse_tes':p['tes_keyword'],
              'smoothing_sigma':0.004,
              'keys_to_pack':keys_to_pack,
              }
    lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
    lfp, rip_sec = lfps[0], rips_sec[0]

    outputs = define_Xb_Tb(lfp, rip_sec, **kwargs)
    print(outputs.keys())
    '''
    samp_rate = kwargs.get('samp_rate', 1000)
    max_seq_len_pts = kwargs.get('max_seq_len_pts', 416)
    n_mouse_tes = kwargs.get('n_mouse_tes')

    dtype = np.float16 if kwargs.get('use_fp16', True) else np.float32

    perturb_pts = np.random.randint(0, max_seq_len_pts) if kwargs.get('use_perturb', False) else 0
    perturb_sec = perturb_pts / samp_rate


    cls_id = {'padding':-10,
              'back_ground':-2,
              'true_ripple':0,
              'false_ripple':1,
               }

    rip_pts = pd.DataFrame({'start_pts':rip_sec['start_sec']*samp_rate,
                            'end_pts':rip_sec['end_sec']*samp_rate,
                            'label':rip_sec['label_cleaned_from_gmm_wo_mouse{}'.format(n_mouse_tes)]
                            })

    rip_pts_arr = np.array(rip_pts)

    last_rip_start_pts = int(rip_pts['start_pts'].iloc[-1])
    lfp = lfp[perturb_pts:last_rip_start_pts]

    step = kwargs.get('step') if kwargs.get('step') else max_seq_len_pts

    # Xb
    slices = util.view_as_windows(lfp, window_shape=(max_seq_len_pts,), step=step)

    # slices time
    slices_start_pts = np.array([perturb_pts + step*i for i in range(len(slices))]) + 1e-10
    # slices_center_pts = slices_start_pts + int(max_seq_len_pts/2)
    slices_end_pts = slices_start_pts + max_seq_len_pts

    # Ripples indices
    on_the_start_rips_indi = np.array([bisect_left(rip_pts_arr[:,0], slices_start_pts[i]) -1 \
                                       for i in range(len(slices))])
    on_the_end_rips_indi = np.array([bisect_right(rip_pts_arr[:,1], slices_end_pts[i]) \
                                     for i in range(len(slices))])

    in_slice_rip_indi = [np.arange(on_the_start_rips_indi[i], on_the_end_rips_indi[i]+1) for i in range(len(slices))]

    # In-slice time
    rips_start_pts_in_slices = [np.clip(rip_pts_arr[:, 0][in_slice_rip_indi[i]] - slices_start_pts[i], 0, None) \
                                for i in range(len(slices))]
    rips_end_pts_in_slices = [np.clip(rip_pts_arr[:, 1][in_slice_rip_indi[i]] - slices_start_pts[i], None, max_seq_len_pts)
                                      for i in range(len(slices))]
    rips_center_pts_in_slices = [np.clip(np.mean([rips_start_pts_in_slices[i], rips_end_pts_in_slices[i]], axis=0).astype(np.int),
                                         0, max_seq_len_pts-1)
                                 for i in range(len(slices))]
    rips_w_pts_in_slices = [np.abs(rips_start_pts_in_slices[i] - rips_end_pts_in_slices[i])
                            for i in range(len(slices))]

    # In-slice Classes
    rips_cls_in_slices = [np.array(rip_pts_arr[:, 2][in_slice_rip_indi[i]]) for i in range(len(slices))]

    # ## Formatting for CenterNet, slow but OK
    # Xb = np.array(slices)
    # Tb_hm, Tb_w = np.zeros_like(Xb).astype(np.float32), np.zeros_like(Xb).astype(np.float32) # Initialization
    # for i_Xb in range(len(Xb)):
    #     centers = np.clip(rips_center_pts_in_slices[i_Xb].astype(np.int), 0, max_seq_len_pts -1)
    #     widths = np.clip(rips_w_pts_in_slices[i_Xb].astype(np.int), 0, max_seq_len_pts -1)
    #     classes = rips_cls_in_slices[i_Xb]
    #     assert len(centers) == len(widths)
    #     assert len(centers) == len(classes)
    #     indi_ripple = (classes == cls_id['true_ripple'])
    #     include_ripple = indi_ripple.any()
    #     if include_ripple:
    #         Tb_hm[i_Xb, centers[indi_ripple]] = 1
    #         Tb_w[i_Xb, centers[indi_ripple]] = widths[indi_ripple]
    #         # Gaussian Smoothing for the heatmap
    #         Tb_hm[i_Xb] = gaussian_smooth(Tb_hm[i_Xb], kwargs['smoothing_sigma'], samp_rate)
    #         Tb_hm[i_Xb] = np.clip((Tb_hm[i_Xb] / Tb_hm[i_Xb].max()), 0, 1)


    ## Formatting for CenterNet, slow but OK, using sparse tensor for reducing the RAM usage
    Xb = np.array(slices)

    # Tb_hm, Tb_w = np.zeros_like(Xb).astype(np.float32), np.zeros_like(Xb).astype(np.int16) # Initialization
    # # Tb_hm, Tb_w
    # i_Xbs = np.hstack([np.ones(len(rips_center_pts_in_slices[i]), dtype=np.int)*i
    #                    for i in range(len(rips_center_pts_in_slices))]).astype(np.int32)
    # i_c = np.hstack(rips_center_pts_in_slices).astype(np.int16)
    # v_c = 1 # torch.ones_like(torch.LongTensor(i_c))
    # v_w = np.hstack(rips_w_pts_in_slices).astype(np.int16)

    # Tb_hm[i_Xbs, i_c] = v_c
    # Tb_w[i_Xbs, i_c] = v_w

    # ## Gaussian Filtering
    # for i in range(len(Tb_hm)):
    #     Tb_hm[i] = gaussian_smooth(Tb_hm[i], kwargs['smoothing_sigma'], samp_rate) if (Tb_hm[i] == 1).any() else Tb_hm[i]
    #     Tb_hm[i] = np.clip((Tb_hm[i] / Tb_hm[i].max()), 0, 1)
    # Tb_hm = Tb_hm.astype(np.float16)


    '''
    i = np.random.randint(len(Tb_hm))
    plt.plot(Xb[i])
    plt.plot(Tb_hm[i]*100)
    plt.plot(Tb_w[i])
    '''

    # outputs = {'Xb':Xb, 'Tb_hm':Tb_hm, 'Tb_w':Tb_w}
    Tb_cls = pad_sequence(rips_cls_in_slices, padding_value=cls_id['back_ground'])
    true_ripple_mask = (Tb_cls == cls_id['true_ripple'])
    Tb_c_pad = pad_sequence(rips_center_pts_in_slices, padding_value=0).astype(np.int32) * true_ripple_mask
    Tb_w_pad = pad_sequence(rips_w_pts_in_slices, padding_value=0).astype(np.int32) * true_ripple_mask

    outputs = {'Xb':Xb,
               'Tb_c_pad':Tb_c_pad,
               'Tb_w_pad':Tb_w_pad}

    if kwargs['use_shuffle']: # 2nd shuffle
      outputs = mf.shuffle_dict(outputs)

    return outputs


def define_Xb_Tb_wrapper(arg_list):
  args, kwargs = arg_list
  return define_Xb_Tb(*args, **kwargs)


def multi_define_Xb_Tb(arg_list):
    '''
    keys_to_pack = ['Xb', 'Tb_c_pad', 'Tb_w_pad']
    kwargs = {'samp_rate':1000,
              'use_fp16':True,
              'use_shuffle':True,
              'max_seq_len_pts':512,
              'step':None,
              'use_perturb':True,
              'define_ripples':True,
              'n_mouse_tes':p['tes_keyword'],
              'smoothing_sigma':0.004,
              'keys_to_pack':keys_to_pack,
              }
    lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
    arg_list = [((lfps[i], rips_sec[i]), kwargs) for i in range(len(lfps))] # args, kwargs
    Xb_Tbs_dict_list = multi_define_Xb_Tb(arg_list)

    print(Xb_Tbs_dict_list[0]['Xb'])

    i = np.random.randint(len(Xb_Tbs_dict_list[0]['Xb']))
    plt.plot(Xb_Tbs_dict_list[0]['Xb'][i])
    plt.plot(Xb_Tbs_dict_list[0]['Tb_hm'][i]*100)
    plt.plot(Xb_Tbs_dict_list[0]['Tb_w'][i])
    '''
    n_cpus = mp.cpu_count() # 20, 10
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
    keys_to_pack = ['Xb', 'Tb_c_pad', 'Tb_w_pad']
    kwargs = {'samp_rate':1000,
              'use_fp16':True,
              'use_shuffle':True,
              'max_seq_len_pts':416,
              'step':None,
              'use_perturb':True,
              'define_ripples':True,
              'n_mouse_tes':p['tes_keyword'],
              'smoothing_sigma':0.004,
              'keys_to_pack':keys_to_pack,
              }
    lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)

    Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs)

    print(Xb_Tbs_dict['Tb_w_pad'])

    i = np.random.randint(len(Xb_Tbs_dict['Xb']))
    plt.plot(Xb_Tbs_dict['Xb'][i])
    # plt.plot(Xb_Tbs_dict['Tb_c_pad'][i]*100)
    # plt.plot(Xb_Tbs_dict['Tb_w'][i])
    '''

    arg_list = [((lfps[i], rips_sec[i]), kwargs) for i in range(len(lfps))] # args, kwargs
    Xb_Tbs_dict_list = multi_define_Xb_Tb(arg_list) # too big

    del arg_list, lfps, rips_sec
    gc.collect()

    def gathers_mp_outputs_dict_list(mp_outputs_dict_list, **kwargs): # Xb_Tb_keys=None
        # mp_outputs_dict_list = Xb_Tbs_dict_list
        mp_output_keys = mp_outputs_dict_list[0].keys()
        n_keys = len(mp_output_keys)
        assert n_keys == len(mp_outputs_dict_list[0])

        gathered_dict = mf.listed_dict(mp_output_keys)
        for i in range(len(mp_outputs_dict_list)):
            for k in mp_output_keys:
                gathered_dict[k].append(mp_outputs_dict_list[i][k])

        # Xb
        gathered_dict['Xb'] = np.concatenate(gathered_dict['Xb'])

        max_n_bounding_ranges_in_an_input = max(np.array([gathered_dict['Tb_c_pad'][i].shape[1]
                                                          for i in range(len(gathered_dict['Tb_c_pad']))]))

        # Tb_c
        dtype = gathered_dict['Tb_c_pad'][0].dtype
        for i in range(len(gathered_dict['Tb_c_pad'])):
            s1, s2 = gathered_dict['Tb_c_pad'][i].shape
            if max_n_bounding_ranges_in_an_input - s2 > 0:
                zeros_arr = (np.zeros([s1, max_n_bounding_ranges_in_an_input - s2])).astype(dtype)
                gathered_dict['Tb_c_pad'][i] = np.concatenate([gathered_dict['Tb_c_pad'][i], zeros_arr], axis=1)
        gathered_dict['Tb_c_pad'] = np.concatenate(gathered_dict['Tb_c_pad'])

        # Tb_w
        dtype = gathered_dict['Tb_w_pad'][0].dtype
        for i in range(len(gathered_dict['Tb_w_pad'])):
            s1, s2 = gathered_dict['Tb_w_pad'][i].shape
            if max_n_bounding_ranges_in_an_input - s2 > 0:
                zeros_arr = (-np.ones([s1, max_n_bounding_ranges_in_an_input - s2])).astype(dtype)
                gathered_dict['Tb_w_pad'][i] = np.concatenate([gathered_dict['Tb_w_pad'][i], zeros_arr], axis=1)
        gathered_dict['Tb_w_pad'] = np.concatenate(gathered_dict['Tb_w_pad'])

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


def to_Tb_hm_and_w(Xb, Tb_c_pad, Tb_w_pad, **kwargs):
    '''
    Tb_hm and Tb_w_pad is the half length of the Xb. (R=2)
    This is because convolusion's shrinks inputs lengths and using padding would cause troubles along time axis.
    '''
    R = kwargs['R']
    mask = (Tb_c_pad != 0)
    i_Xbs = torch.arange(len(Tb_c_pad)).unsqueeze(-1).repeat(1, Tb_c_pad.size(1))[mask]
    i_c = Tb_c_pad[mask]
    v_c = torch.ones_like(torch.LongTensor(i_c.long()))
    v_w = Tb_w_pad[mask]

    ## High Resolusion
    Tb_hm = np.zeros_like(Xb).astype(np.float32) # Initialization
    Tb_w = np.zeros_like(Xb).astype(np.int16)
    Tb_hm[i_Xbs.long(), i_c.long()] = v_c
    Tb_w[i_Xbs.long(), i_c.long()] = v_w
    # Gaussian Filtering
    for i in range(len(Tb_hm)):
        ripple_included = (Tb_hm[i] == 1).any()
        if ripple_included:
            Tb_hm[i] = gaussian_smooth(Tb_hm[i], kwargs['smoothing_sigma'], kwargs['samp_rate'])
            Tb_hm[i] = Tb_hm[i] / Tb_hm[i].max()
            Tb_hm[i] = np.clip(Tb_hm[i], 0, 1)
    Tb_hm = Tb_hm.astype(np.float16)
    # To Torch.Tensor
    Tb_hm, Tb_w = torch.tensor(Tb_hm), torch.tensor(Tb_w)

    ## Low Resolution
    i_c_R, v_w_R, v_c_R = i_c/R, v_w/R, v_c
    Tb_hm_R = np.zeros((Xb.shape[0], int(Xb.shape[1]/R)), dtype=np.float32) # Initialization
    Tb_w_R = np.zeros((Xb.shape[0], int(Xb.shape[1]/R)), dtype=np.int16)
    Tb_hm_R[i_Xbs.long(), i_c_R.long()] = v_c_R
    Tb_w_R[i_Xbs.long(), i_c_R.long()] = v_w_R
    # Gaussian Filtering
    for i in range(len(Tb_hm_R)):
        ripple_included = (Tb_hm_R[i] == 1).any()
        if ripple_included:
            Tb_hm_R[i] = gaussian_smooth(Tb_hm_R[i], kwargs['smoothing_sigma']/R, kwargs['samp_rate'])
            Tb_hm_R[i] = Tb_hm_R[i] / Tb_hm_R[i].max()
            Tb_hm_R[i] = np.clip(Tb_hm_R[i], 0, 1)
    Tb_hm_R = Tb_hm_R.astype(np.float16)
    # To Torch.Tensor
    Tb_hm_R, Tb_w_R = torch.tensor(Tb_hm_R), torch.tensor(Tb_w_R)

    return Tb_hm, Tb_w, Tb_hm_R, Tb_w_R

def normalize_Xb(Xb):
    dtype = Xb.dtype
    Xb = Xb.float()
    mean, std = Xb.mean(dim=1, keepdim=True), Xb.std(dim=1, keepdim=True)
    Xb = (Xb - mean) / std
    return Xb.to(dtype)


def pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs):
    '''
    keys_to_pack = ['Xb', 'Tb_c_pad', 'Tb_w_pad']
    kwargs = {'samp_rate':1000,
              'use_fp16':True,
              'use_shuffle':True,
              'max_seq_len_pts':416,
              'step':None,
              'use_perturb':True,
              'keys_to_pack':keys_to_pack,
              'bs':64,
              'nw':10,
              'pm':False, # RuntimeError: Caught RuntimeError in pin memory thread for device 0.
              'drop_last':True,
              'n_mouse_tes':p['tes_keyword'],
              'smoothing_sigma':0.004,
              'R':2,
              }

    lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
    Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs)

    dataloader = pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs)
    batch = next(iter(dataloader))

    Xb, Tb_hm, Tb_w, Tb_hm_R, Tb_w_R = batch

    i = np.random.randint(len(Xb))
    plt.plot(Xb[i])
    plt.plot(Tb_hm[i].float()*1)
    plt.plot(Tb_w[i].float())

    '''
    # dataset = utils.TensorDataset(*(Xb_Tbs_dict[k] for k in kwargs['keys_to_pack']))
    dataset = utils.TensorDataset(*(torch.tensor(Xb_Tbs_dict[k]) for k in kwargs['keys_to_pack']))

    def collate_fn(batch, **kwargs):
          '''
          Transform Tb_CenterX_W and Tb_cls for YOLO.
          '''
          Xb = torch.stack([item[0] for item in batch], dim=0)
          Tb_c_pad = torch.stack([item[1] for item in batch], dim=0)
          Tb_w_pad = torch.stack([item[2] for item in batch], dim=0)

          # Clear padding and conduct gaussian filtering
          Tb_hm, Tb_w, Tb_hm_R, Tb_w_R = to_Tb_hm_and_w(Xb, Tb_c_pad, Tb_w_pad, **kwargs)

          # Normalize Samples
          Xb = normalize_Xb(Xb)
          return Xb, Tb_hm, Tb_w, Tb_hm_R, Tb_w_R

    class collate_fn_class():
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, batch):
            return collate_fn(batch, **self.kwargs)

    collate_fn_instance = collate_fn_class(**kwargs)

    dataloader = utils.DataLoader(dataset,
                                  batch_size=kwargs['bs'],
                                  shuffle=kwargs['use_shuffle'],
                                  num_workers=kwargs['nw'],
                                  pin_memory=kwargs['pm'],
                                  drop_last=kwargs['drop_last'],
                                  collate_fn=collate_fn_instance,
                                  )

    return dataloader


def mk_dataloader(lfp_fpaths, **kwargs):
    '''
    keys_to_pack = ['Xb', 'Tb_c_pad', 'Tb_w_pad']
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
              'n_mouse_tes':p['tes_keyword'],
              'smoothing_sigma':0.004,
              'R':2,
              }

    dataloader = mk_dataloader(p['fpaths_tra'], **kwargs)# pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs)
    batch = next(iter(dataloader))
    Xb, Tb_hm, Tb_w = batch
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
    keys_to_pack = ['Xb', 'Tb_c_pad', 'Tb_w_pad']
    kwargs = {'samp_rate':1000,
              'use_fp16':True,
              'use_shuffle':True,
              'max_seq_len_pts':992,
              'step':None,
              'use_perturb':True,
              'define_ripples':True,
              'keys_to_pack':keys_to_pack,
              'bs':64,
              'nw':20,
              'pm':True,
              'drop_last':True,
              'n_mouse_tes':p['tes_keyword'],
              'smoothing_sigma':0.004,
              'R':2,
              }

    lfp_fpaths = p['fpaths_tra']
    dl_fulf = DataloaderFulfiller(lfp_fpaths, **kwargs)

    n_samples = dl_fulf.get_n_samples()
    print(n_samples)

    dl = dl_fulf()
    batch = next(iter(dl))
    Xb, Tb_hm, Tb_w, Tb_hm_R, Tb_w_R = batch

    i = np.random.randint(len(Xb))
    plt.plot(Xb[i])
    plt.plot(Tb_hm[i].float())
    plt.plot(Tb_w[i].float()*10)


    _check_samples(Xb, targets[:,:-1], label_conf=targets[:,-1], max_plot=10)
    '''
    def __init__(self, lfp_fpaths, n_use_lfp=184, **kwargs):
        if n_use_lfp < len(lfp_fpaths):
            print('--- Developping --- \nOnly {} LFPs were randomly chosen.'.format(n_use_lfp))
            indi_use_lfp = np.random.permutation(len(lfp_fpaths))[:n_use_lfp]
            lfp_fpaths = list(np.array(lfp_fpaths)[indi_use_lfp])
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


if __name__ == '__main__':
    ## Load Paths
    p = mf.listed_dict()
    loadpath_npy_list = '../data/1kHz_npy_list.pkl' # '../data/2kHz_npy_list.pkl'
    p['n_load_all'] = 12 #12 # args.n_load_all # 186   # 12 -> fast, 176 -> w/o 05/day5, 186 -> full
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

    keys_to_pack = ['Xb', 'Tb_hm', 'Tb_w']
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
              'n_mouse_tes':p['tes_keyword'],
              'smoothing_sigma':0.004,
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
