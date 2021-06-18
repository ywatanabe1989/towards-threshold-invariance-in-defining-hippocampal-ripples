import sys
sys.path.append('./')
import myutils.myfunc as mf
sys.path.append('./06_File_IO')
sys.path.append('./07_Learning/')
import os

from bisect import bisect_left
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


# ##################################################
# ## Load Paths
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
# ##################################################



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
  kwargs = {'samp_rate':1000}
  cvt_lpath_lfp_2_lpath_rip(p['fpaths_tra'][0], **kwargs)
  '''
  samp_rate = glob_samp_rate(lpath_lfp)
  lsamp_str = cvt_samp_rate_int2str(**kwargs)
  lpath_rip = lpath_lfp.replace(lsamp_str, '1kHz').replace('.npy', '_riptimes_levels.pkl')
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
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tes'], **kwargs)
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
  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':200,
            'step':None,
            'use_perturb':False,
            'max_distancems':None,
            'ripples_binary_classification':False,
            'ripples_multi_classification':False,
            'estimates_ripple_params':True,
            }
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tes'], **kwargs)
  lfp, rip_sec = lfps[0], rips_sec[0]
  outputs = define_Xb_Tb(lfp, rip_sec, **kwargs)
  print(np.unique(outputs['Tb_label']))
  print(outputs.keys())
  '''
  samp_rate = kwargs.get('samp_rate', 1000)
  max_seq_len_pts = kwargs.get('max_seq_len_pts', 200)
  # max_distance_ms = kwargs.get('max_distance_ms') if kwargs.get('max_distance_ms') else int(max_seq_len_pts/2)
  dtype = np.float16 if kwargs.get('use_fp16', True) else np.float32

  perturb_pts = np.random.randint(0, max_seq_len_pts) if kwargs.get('use_perturb', False) else 0
  perturb_sec = perturb_pts / samp_rate

  rip_sec_cp = rip_sec.copy()

  the_4th_to_last_rip_end_pts = int(rip_sec_cp.iloc[-4]['end_sec'] * samp_rate)
  lfp = lfp[perturb_pts:the_4th_to_last_rip_end_pts]

  step = kwargs.get('step') if kwargs.get('step') else max_seq_len_pts

  # Xb
  slices = util.view_as_windows(lfp, window_shape=(max_seq_len_pts,), step=step)

  slices_start_pts = np.array([perturb_pts + step*i for i in range(len(slices))]) + 1e-10
  slices_start_sec = slices_start_pts / samp_rate

  slices_center_pts = slices_start_pts + int(max_seq_len_pts/2)
  slices_center_sec = slices_center_pts / samp_rate

  slices_end_pts = slices_start_pts + max_seq_len_pts
  slices_end_sec = slices_end_pts / samp_rate

  the_1st_rips_indi = np.array([bisect_left(rip_sec_cp['start_sec'].values, slices_start_sec[i]) -1 for i in range(len(slices))])
  the_2nd_rips_indi = the_1st_rips_indi + 1
  the_3rd_rips_indi = the_1st_rips_indi + 2
  # next_rip_indi_list = [bisect_left(rip_sec_cp['start_sec'].values, slices_end_sec[i]) for i in range(len(slices_end_sec))]

  the_1st_rips_level = rip_sec_cp.iloc[the_1st_rips_indi]['level'].to_numpy()
  the_2nd_rips_level = rip_sec_cp.iloc[the_2nd_rips_indi]['level'].to_numpy()
  the_3rd_rips_level = rip_sec_cp.iloc[the_3rd_rips_indi]['level'].to_numpy() # IndexError: positional indexers are out-of-bounds

  are_the_1st_rips_over_the_slice_end = (slices_end_sec < rip_sec_cp.iloc[the_1st_rips_indi]['end_sec']).to_numpy()
  are_the_2nd_rips_over_the_slice_end = (slices_end_sec < rip_sec_cp.iloc[the_2nd_rips_indi]['end_sec']).to_numpy()
  are_the_3rd_rips_over_the_slice_end = (slices_end_sec < rip_sec_cp.iloc[the_3rd_rips_indi]['end_sec']).to_numpy()

  # Tb_label
  are_label_0 = (the_1st_rips_level == 0) * are_the_1st_rips_over_the_slice_end

  labels_k = (the_1st_rips_level == 0) * ~are_the_1st_rips_over_the_slice_end \
           * the_2nd_rips_level * ~are_the_2nd_rips_over_the_slice_end \
           * (the_3rd_rips_level == 0) * are_the_3rd_rips_over_the_slice_end

  Tb_label = np.ones(len(slices), dtype=np.int)*(-1) # initialize
  Tb_label[are_label_0] = 0
  for k in [1, 2, 3, 4, 5]:
    Tb_label[labels_k == k] = k

  '''
  ## Confirm Tb_label
  # 1) print numbers
  i = np.random.randint(1e5)
  print('slices start: {}'.format(slices_start_sec[i:i+9]))
  print('slices end  : {}'.format(slices_end_sec[i:i+9]))
  print('slices label: {}'.format(Tb_label[i:i+9]))
  print(rip_sec[(slices_start_sec[i] < rip_sec['end_sec']) & (rip_sec['start_sec'] < slices_end_sec[i+9])])

  # 2) plot # fixme
  i = np.random.randint(1e2)
  n_slices = 10
  _start_sec, _end_sec = slices_start_sec[i], slices_end_sec[i+n_slices]
  _start_pts, _end_pts = int(_start_sec*samp_rate), int(_end_sec*samp_rate)
  _lfp = lfp[_start_pts:_end_pts]
  _rip_sec = rip_sec_cp[(_start_sec < rip_sec_cp['start_sec']) & (rip_sec_cp['end_sec'] < _end_sec)] # fixme
  _slices = slices[i:i+n_slices]
  _slices_start_sec = slices_start_sec[i:i+n_slices]
  _slices_end_sec = slices_end_sec[i:i+n_slices]

  t = 1.*np.arange(_start_sec, _end_sec, 1./samp_rate)
  plt.plot(t, _lfp)

  for rip in _rip_sec.itertuples():
    # ax.axvspan(rip.start_sec, ax.end_sec, alpha=0.3, color='red')
    t_rip = 1.*np.arange(rip.start_sec, rip.end_sec, 1./samp_rate)
    plt.plot(t_rip, np.ones_like(t_rip)*rip.level*100, color='red')

  for i_slice in range(n_slices):
    t_slice = 1.*np.arange(_slices_start_sec[i_slice], _slices_end_sec[i_slice], 1./samp_rate)
    plt.plot(t_slice, np.ones_like(t_slice)*Tb_label[i_slice].numpy()*100, color='green')
  '''

  assert kwargs['ripples_binary_classification'] or \
         kwargs['ripples_multi_classification'] or \
         kwargs['estimates_ripple_params']

  if kwargs['estimates_ripple_params']:
    '''
    In this case, slices which don't have ripple parameters (= level 0 ripples) have to be excluded.
    '''
    are_label_k = (Tb_label > 0)
    slices_label_k = slices[are_label_k]
    slices_start_sec_label_k = slices_start_sec[are_label_k]
    Tb_label_k = Tb_label[are_label_k]
    Xb = slices_label_k

    # Fetches Ripple Prameters
    rips_indi_of_label_k_slices = the_2nd_rips_indi[are_label_k]
    rips_params_of_label_k_slices = rip_sec.iloc[rips_indi_of_label_k_slices]

    # pd.DataFrame to Dictionary
    _outputs = rips_params_of_label_k_slices.to_dict(orient='list')
    keys = _outputs.keys()
    for k in keys:
      _outputs[k] = np.array(_outputs[k]) if not '_sec' in k else np.array(_outputs[k]) - slices_start_sec_label_k

    outputs = {'Xb':Xb, 'Tb_label':Tb_label_k}
    outputs.update(_outputs)


  else:
    # Excludes undefined slices
    are_defined = (Tb_label != -1)
    Xb = slices[are_defined]
    Tb_label = Tb_label[are_defined]
    assert len(Xb) == len(Tb_label)

    if kwargs['ripples_multi_classification']:
      outputs = {'Xb':Xb, 'Tb_label':Tb_label}

    if kwargs['ripples_binary_classification']:
      are_label_0_or_5 = (Tb_label == 0) + (Tb_label == 5)
      Xb = Xb[are_label_0_or_5] # exclude label 1,2,3,4
      Tb_label = Tb_label[are_label_0_or_5] # exclude label 1,2,3,4
      Tb_label[Tb_label == 5] = 1 # convert label from 5 to 1 for naturally applying classification loss
      assert len(Xb) == len(Tb_label)
      outputs = {'Xb':Xb, 'Tb_label':Tb_label}

  '''
  # Check the outputs' lengthes
  keys = outputs.keys()
  first = True
  for k in keys:
    if first:
      length = len(outputs[k])
      first = False
    if not first:
      length_next = len(outputs[k])
      assert length_next == length
      length_next = length
  '''

  if kwargs['use_shuffle']: # 2nd shuffle
    outputs = mf.shuffle_dict(outputs)

  return outputs


def define_Xb_Tb_wrapper(arg_list):
  args, kwargs = arg_list
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
  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':200,
            'step':None,
            'use_perturb':True,
            'ripples_binary_classification':False,
            'ripples_multi_classification':False,
            'estimates_ripple_params':True,
            }
  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
  Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs)

  # Confirm with plotting
  plot_label = 5
  max_plot = 1
  plot = 0
  for _ in range(1000):
    i = np.random.randint(1e3)
    label = int(Xb_Tbs_dict['Tb_label'][i])

    if label == plot_label:
      fig, ax = plt.subplots()
      ax.plot(np.arange(200)/kwargs['samp_rate'], Xb_Tbs_dict['Xb'][i])

      try: # when 'estimates_ripple_params':True,
        txt = 'start_sec: {:2f} \n\
               end_sec: {:2f} \n\
               peak_posi_sec : {:2f} \n\
               relat_peak_posi : {:2f} \n\
               ave_power: {:2f} \n\
               peak_power: {:2f} \n\
               peak_freq: {:2f} \n\
               gamma_ave_power: {:2f}'\
              .format(Xb_Tbs_dict['start_sec'][i],
                      Xb_Tbs_dict['end_sec'][i],
                      Xb_Tbs_dict['ripple_peak_posi_sec'][i],
                      Xb_Tbs_dict['ripple_relat_peak_posi'][i],
                      Xb_Tbs_dict['ripple_ave_power'][i],
                      Xb_Tbs_dict['ripple_peak_power'][i],
                      Xb_Tbs_dict['ripple_peak_frequency_hz'][i],
                      Xb_Tbs_dict['gamma_ave_power'][i]
                      )
      except:
        txt = None

      ax.text(0,0, txt, transform=ax.transAxes)
      plt.title('Label {}'.format(label))
      plot += 1
      if plot == max_plot:
        break
  '''

  arg_list = [((lfps[i], rips_sec[i]), kwargs) for i in range(len(lfps))] # args, kwargs

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
  keys_to_pack_binary_classification = ['Xb', 'Tb_label']
  keys_to_pack_multi_classification = ['Xb', 'Tb_label']
  keys_to_pack_estimates_ripple_params = ['Xb',
                                          'Tb_label',
                                          'start_sec',
                                          'end_sec',
                                          'ripple_peak_posi_sec',
                                          'ripple_relat_peak_posi',
                                          'ripple_ave_power',
                                          'ripple_peak_power',
                                          'ripple_peak_frequency_hz',
                                          'gamma_ave_power',]

  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':200,
            'step':None,
            'use_perturb':True,
            'ripples_binary_classification':True,
            'ripples_multi_classification':False,
            'estimates_ripple_params':False,
            'bs':64,
            'nw':10,
            'pm':True,
            'drop_last':True,
            'collate_fn_class':None,
            'keys_to_pack_binary_classification':keys_to_pack_binary_classification,
            'keys_to_pack_multi_classification':keys_to_pack_multi_classification,
            'keys_to_pack_estimates_ripple_params':keys_to_pack_estimates_ripple_params,
            }

  if kwargs['ripples_binary_classification']:
    kwargs['keys_to_pack'] = kwargs['keys_to_pack_binary_classification']
  elif kwargs['ripples_multi_classification']:
    kwargs['keys_to_pack'] = kwargs['keys_to_pack_multi_classification']
  elif kwargs['estimates_ripple_params']:
    kwargs['keys_to_pack'] = kwargs['keys_to_pack_estimates_ripple_params']

  lfps, rips_sec = load_lfps_rips_sec(p['fpaths_tra'], **kwargs)
  Xb_Tbs_dict = multi_define_Xb_Tb_wrapper(lfps, rips_sec, **kwargs)
  dataloader = pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs)
  next(iter(dataloader))
  '''

  dataset = utils.TensorDataset(*(torch.tensor(Xb_Tbs_dict[k]) for k in kwargs['keys_to_pack']))

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
  keys_to_pack_binary_classification = ['Xb', 'Tb_label']
  keys_to_pack_multi_classification = ['Xb', 'Tb_label']
  keys_to_pack_estimates_ripple_params = ['Xb',
                                          'Tb_label',
                                          'start_sec',
                                          'end_sec',
                                          'ripple_peak_posi_sec',
                                          'ripple_relat_peak_posi',
                                          'ripple_ave_power',
                                          'ripple_peak_power',
                                          'ripple_peak_frequency_hz',
                                          'gamma_ave_power',]

  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':200,
            'step':None,
            'use_perturb':True,
            'ripples_binary_classification':False,
            'ripples_multi_classification':False,
            'estimates_ripple_params':True,
            'bs':64,
            'nw':10,
            'pm':True,
            'drop_last':True,
            'collate_fn_class':None,
            'keys_to_pack_binary_classification':keys_to_pack_binary_classification,
            'keys_to_pack_multi_classification':keys_to_pack_multi_classification,
            'keys_to_pack_estimates_ripple_params':keys_to_pack_estimates_ripple_params,
            }

  if kwargs['ripples_binary_classification']:
    kwargs['keys_to_pack'] = kwargs['keys_to_pack_binary_classification']
  elif kwargs['ripples_multi_classification']:
    kwargs['keys_to_pack'] = kwargs['keys_to_pack_multi_classification']
  elif kwargs['estimates_ripple_params']:
    kwargs['keys_to_pack'] = kwargs['keys_to_pack_estimates_ripple_params']

  dataloader = mk_dataloader(p['fpaths_tra'], **kwargs)# pack_Xb_Tb_to_dataloader(Xb_Tbs_dict, **kwargs)
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
  keys_to_pack_binary_classification = ['Xb', 'Tb_label']
  keys_to_pack_multi_classification = ['Xb', 'Tb_label']
  keys_to_pack_estimates_ripple_params = ['Xb',
                                          'Tb_label',
                                          'start_sec',
                                          'end_sec',
                                          'ripple_peak_posi_sec',
                                          'ripple_relat_peak_posi',
                                          'ripple_ave_power',
                                          'ripple_peak_power',
                                          'ripple_peak_frequency_hz',
                                          'gamma_ave_power',]

  kwargs = {'samp_rate':1000,
            'use_fp16':True,
            'use_shuffle':True,
            'max_seq_len_pts':200,
            'step':None,
            'use_perturb':True,
            'ripples_binary_classification':False,
            'ripples_multi_classification':False,
            'estimates_ripple_params':True,
            'bs':64,
            'nw':10,
            'pm':True,
            'drop_last':True,
            'collate_fn_class':None,
            'keys_to_pack_binary_classification':keys_to_pack_binary_classification,
            'keys_to_pack_multi_classification':keys_to_pack_multi_classification,
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
        # Switches keys_to_pack
    if kwargs['ripples_binary_classification']:
      self.kwargs['keys_to_pack'] = kwargs['keys_to_pack_binary_classification']
    elif kwargs['ripples_multi_classification']:
      self.kwargs['keys_to_pack'] = kwargs['keys_to_pack_multi_classification']
    elif kwargs['estimates_ripple_params']:
      self.kwargs['keys_to_pack'] = kwargs['keys_to_pack_estimates_ripple_params']

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
