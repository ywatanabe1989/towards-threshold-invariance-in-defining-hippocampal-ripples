### my own packages
import sys
sys.path.append('./')
sys.path.append('./myutils')
import myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_191020 import dataloader_fulfiller
sys.path.append('./07_Learning/')
from balance_xentropy_loss import BalanceCrossEntropyLoss
from apex import amp
sys.path.append('./11_Models/')
from model_191015 import Model
sys.path.append('./10_Evaluation/')
from glob_the_best_model_dir import glob_the_best_model_dir

### others
import argparse
import os
import datetime
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
from pprint import pprint
from sklearn.utils import shuffle


def load_params(dirpath, plot=False):
  global timer, d, p, m, weight_path, datadir

  print('Data directory: {}'.format(dirpath))
  d = mf.pkl_load(dirpath + 'data.pkl')
  p = mf.pkl_load(dirpath + 'params.pkl')
  m = mf.pkl_load(dirpath + 'model.pkl')
  # m['use_opti'] = 'sgd' # fixme
  p['bs_tes'] = 1024 * 10 # fixme
  weight_path = dirpath + 'weight.pth'


def init_dataloader():
  global dl_fulf_tes
  keys_to_pack_binary_classification = ['Xb', 'Tb_label']
  keys_to_pack_multi_classification = ['Xb', 'Tb_SD']
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

  kwargs_dl_tes = {'ripples_binary_classification':True,
                   'ripples_multi_classification':False,
                   'estimates_ripple_params':False,
                   'samp_rate':p['samp_rate'],
                   'use_fp16':m['use_fp16'],
                   'use_shuffle':False, # Note
                   'max_seq_len_pts':p['max_seq_len'],
                   'step':None,
                   'use_perturb':True, # Note
                   'bs':p['bs_tes'],
                   'nw':10,
                   'pm':True,
                   'drop_last':True,
                   'collate_fn_class':None,
                   'keys_to_pack_binary_classification':keys_to_pack_binary_classification,
                   'keys_to_pack_multi_classification':keys_to_pack_multi_classification,
                   'keys_to_pack_estimates_ripple_params':keys_to_pack_estimates_ripple_params,
                   }

  dl_fulf_tes = dataloader_fulfiller(p['fpaths_tes'], **kwargs_dl_tes) # fixme
  d['n_samples_tes'] = dl_fulf_tes.get_n_samples()
  d['n_batches_per_epoch_tes'] = math.ceil(d['n_samples_tes'] / p['bs_tes'])




def init_ripple_detector_NN():
  global model

  model = Model(input_size=m['n_features'],
                output_size=m['n_out'],
                max_seq_len=p['max_seq_len'],
                samp_rate=p['samp_rate'],
                hidden_size=m['hidden_size'],
                num_layers=m['num_layers'],
                dropout_rnn=m['dropout_rnn'],
                dropout_fc=m['dropout_fc'],
                bidirectional=m['bidirectional'],
                use_input_bn=m['use_input_bn'],
                use_rnn=('rnn' in m['ftr_extractor']),
                use_cnn=('cnn' in m['ftr_extractor']),
                use_transformer=('att' in m['ftr_extractor']),
                transformer_d_model = m['transformer_d_model'],
                use_fft=('fft' in m['ftr_extractor']),
                use_wavelet_scat=('wlt' in m['ftr_extractor']),
                rnn_archi=m['rnn_archi'],
                cnn_n_filter_1st=m['cnn_n_filter_1st'],
                use_activation=m['use_activation'],
                use_loss=m['use_loss'],
                ).to(m['device'])

  if m['use_multitaskloss']:
    global multitaskloss
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
    if m['use_opti'] == 'ranger':
      optimizer = Ranger(learnable_params, lr=m['init_lr'], eps=adam_eps)
    if m['use_opti'] == 'sgd':
      optimizer = torch.optim.SGD(learnable_params, lr=m['init_lr'], nesterov=False)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=num_losses)
  else:
    adam_eps = 1e-8
    if m['use_opti'] == 'ranger':
      optimizer = Ranger(learnable_params, lr=m['init_lr'], eps=adam_eps)
    if m['use_opti'] == 'sgd':
      optimizer = torch.optim.SGD(learnable_params, lr=m['init_lr'], nesterov=False)

  if m['n_gpus'] > 1:
    model = torch.nn.DataParallel(model).to(m['device'])

  if m['use_loss'] == 'bce':
    global sigmoid, bce_criterion
    sigmoid = torch.nn.Sigmoid()
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

  if m['use_loss'] == 'xentropy':
    global softmax, xentropy_criterion
    softmax = torch.nn.Softmax(dim=-1)
    xentropy_criterion = torch.nn.CrossEntropyLoss(reduction='none')

  if m['use_loss'] == 'kldiv':
    global log_softmax, kldiv_criterion
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    kldiv_criterion = torch.nn.KLDivLoss(reduction='none')

  if m['use_loss'] == 'focal':
    global focal_criterion # sigmoid
    sigmoid = torch.nn.Sigmoid()
    focal_criterion = FocalLoss(gamma=10., alpha=0.5) # gamma=2. , alpha=.25

  global balance_loss
  balance_loss = BalanceCrossEntropyLoss(m['n_out'])

def NN_to_eval_mode():
  ## Load weight
  model_state_dict = torch.load(weight_path)
  model.load_state_dict(model_state_dict)
  model.eval()

  if m['use_multitaskloss']:
    multitaskloss.eval()


def _print_log_tes(cum_dict):
  print('-----------------------------------------')
  print('\nTest Epoch: {}'.format(d['epoch'][-1]))
  pprint(cum_dict)
  print('-----------------------------------------')


def test(n_batches=10):
  global cls_container
  d['epoch'] = [1] if not d.get('epoch') else d.get('epoch')
  timer('Epoch {} Test starts'.format(d['epoch'][-1]))
  dl_tes = dl_fulf_tes.fulfill()
  model.eval()
  _ = multitaskloss.eval() if m['use_multitaskloss'] else 0
  p['n_tes'] = len(dl_tes.dataset)

  keys_cum = ['cum_loss_isRipple_tes', 'cum_loss_tot_tes', 'correct_isRipple_tes', 'n_counter_tes',
              'cm_isRipple_tes', 'cr_isRipple_tes']
  cum_dict = mf.init_dict(keys_cum)

  with torch.no_grad():
    for i_batch, batch in enumerate(dl_tes):
      # batch = next(iter(dl_tes)) # if not 'batch' in locals() else batch # for developping purpose
      assert len(batch) == len(dl_fulf_tes.kwargs['keys_to_pack'])

      # Xb
      Xb_Tbs_dict = mf.init_dict(keys=dl_fulf_tes.kwargs['keys_to_pack'], values=batch) # Xb, Tb_dur, Tb_lat = batch

      # to GPU
      for k in dl_fulf_tes.kwargs['keys_to_pack']:
        Xb_Tbs_dict[k] = Xb_Tbs_dict[k].to(m['device'])

      # Model Outputs
      Xb = Xb_Tbs_dict['Xb'] # fixme
      pred_isRipple_logits = model(Xb.unsqueeze(-1))

      # losses
      if m['use_loss'] == 'bce':
        Tb_isRipple = Xb_Tbs_dict['Tb_label'].to(torch.float)
        loss_isRipple = bce_criterion(pred_isRipple_logits, Tb_isRipple.to(torch.float).unsqueeze(-1)).squeeze() # not probs but logits
        pred_isRipple_probs = sigmoid(pred_isRipple_logits)
        pred_isRipple_cls = (pred_isRipple_probs > .5)
        cls_rec.add_target(Tb_isRipple.detach().cpu().numpy())
        cls_rec.add_output_cls(pred_isRipple_cls.detach().cpu().numpy())
        cum_dict['ave_TP_samples\'_prob_tes'] = pred_isRipple_probs[Tb_isRipple.to(torch.bool)].mean().detach().cpu().numpy()

      if m['use_loss'] == 'xentropy':
        Tb_isRipple = Xb_Tbs_dict['Tb_label'].to(torch.long)
        loss_isRipple = xentropy_criterion(pred_isRipple_logits, Tb_isRipple.to(torch.long)) # fixme: Tb_isRipple
        pred_isRipple_probs = softmax(pred_isRipple_logits)
        pred_isRipple_cls = torch.argmax(pred_isRipple_probs, dim=-1)
        cls_rec.add_target(Tb_isRipple.detach().cpu().numpy())
        cls_rec.add_output_cls(pred_isRipple_cls.detach().cpu().numpy())

      if m['use_loss'] == 'focal':
        Tb_isRipple = Xb_Tbs_dict['Tb_label'].to(torch.long)
        loss_isRipple = focal_criterion(pred_isRipple_logits.squeeze(), Tb_isRipple.to(torch.float)) # fixme: Tb_isRipple
        pred_isRipple_probs = sigmoid(pred_isRipple_logits)
        pred_isRipple_cls = (pred_isRipple_probs > .5)
        correct_isRipple = (pred_isRipple_cls == Tb_isRipple.to(torch.long))
        cum_dict['correct_isRipple_tra'] += correct_isRipple.sum().detach().cpu().numpy()

      if m['use_loss'] == 'kldiv':
        Tb_distances_ms_onehot = cvt_Tb_distance_ms(Tb_distances_ms.clone(), p['max_distance_ms'])
        pred_isRipple_logprobs = log_softmax(pred_isRipple_logits)
        pred_isRipple_probs = pred_isRipple_logprobs.exp()
        loss_isRipple = kldiv_criterion(pred_isRipple_logprobs, Tb_distances_ms_onehot.to(m['device'])).sum(dim=-1)

      if i_batch == n_batches:
        break



if __name__ == '__main__':
  ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  ap.add_argument("-nmt", "--n_mouse_tes", default=1, choices=[1,2,3,4,5], type=int, help = "") #  '191022/230633'
  ap.add_argument("-save", action='store_true', help = "")
  ap.add_argument("-plot", action='store_true', help = "")
  args = ap.parse_args()

  ## Load
  timestamps = {
  'Test Mouse Number 01':'191217/170904',
  'Test Mouse Number 02':'191217/170931',
  'Test Mouse Number 03':'191217/170949',
  'Test Mouse Number 04':'191218/020124',
  'Test Mouse Number 05':'191217/171009',
  }
  ts = timestamps['Test Mouse Number 0{}'.format(args.n_mouse_tes)]

  dirpath_root = '../results/' + ts + '/' # fixme
  dirpaths = mf.natsorted_glob(dirpath_root + 'epoch_*/batch_*/')
  dirpath_last = dirpaths[-1]

  load_params(dirpath_last, plot=args.plot)

  '''
  plt.plot(d['ave_loss_isRipple_tra'])
  pprint(d['conf_mat_tra'])
  '''

  init_dataloader()
  init_ripple_detector_NN()
  NN_to_eval_mode()
  global cls_rec
  cls_rec = mf.class_recorder()
  test(n_batches=d['n_batches_per_epoch_tes']) # 100
  conf_mat_tes, cls_report_tes = cls_rec.report(labels_cm=['noRipple', 'Ripple'], labels_cr=[0, 1])
  pprint(conf_mat_tes)
  pprint(cls_report_tes)

  ## Save
  if args.save:
    conf_mat_tes.to_csv(datadir + 'conf_mat_tes.csv')
    cls_report_tes.to_csv(datadir + 'cls_report_tes.csv')

  '''
  import pandas as pd
  from pprint import pprint

  datadirs_on_the_way = {
  'Test Mouse Number 01':'/mnt/nvme/Ripple_Detection/results/191023/212438/epoch_3/batch_16900/',
  'Test Mouse Number 02':'/mnt/nvme/Ripple_Detection/results/191024/132223/epoch_2/batch_0/',
  'Test Mouse Number 03':'/mnt/nvme/Ripple_Detection/results/191025/004805/epoch_3/batch_0/',
  'Test Mouse Number 04':'/mnt/nvme/Ripple_Detection/results/191025/132659/epoch_3/batch_0/',
  'Test Mouse Number 05':'/mnt/nvme/Ripple_Detection/results/191026/011033/epoch_3/batch_0/',
  }

  datadirs_on_the_last = {
  'Test Mouse Number 01':'/mnt/nvme/Ripple_Detection/results/191023/212438/epoch_3/batch_22800/',
  'Test Mouse Number 02':'/mnt/nvme/Ripple_Detection/results/191024/132223/epoch_3/batch_17600/',
  'Test Mouse Number 03':'/mnt/nvme/Ripple_Detection/results/191025/004805/epoch_3/batch_18600/',
  'Test Mouse Number 04':'/mnt/nvme/Ripple_Detection/results/191025/132659/epoch_3/batch_17600/',
  'Test Mouse Number 05':'/mnt/nvme/Ripple_Detection/results/191026/011033/epoch_3/batch_18900/',
  }

  conf_mats_on_the_way = []
  cls_reports_on_the_way = []
  dfs_on_the_way = []
  for i_mouse, k_dict in enumerate(list(datadirs_on_the_way.keys())):
    print(i_mouse, k_dict)
    lpath_conf_mat = datadirs_on_the_way[k_dict] + 'conf_mat_tes.csv'
    lpath_cls_report = datadirs_on_the_way[k_dict] + 'cls_report_tes.csv'
    conf_mats_on_the_way.append(pd.read_csv(lpath_conf_mat).rename(columns={"Unnamed: 0": "Mouse#0{}".format(i_mouse+1)}))
    cls_reports_on_the_way.append(pd.read_csv(lpath_cls_report).rename(columns={"Unnamed: 0": "Mouse#0{}".format(i_mouse+1)}))

  conf_mats_on_the_way = pd.concat(conf_mats_on_the_way, axis=1)
  cls_reports_on_the_way = pd.concat(cls_reports_on_the_way, axis=1)
  pprint(conf_mats_on_the_way)
  pprint(cls_reports_on_the_way)



  conf_mats_on_the_last = []
  cls_reports_on_the_last = []
  dfs_on_the_last = []
  for i_mouse, k_dict in enumerate(list(datadirs_on_the_last.keys())):
    print(i_mouse, k_dict)
    lpath_conf_mat = datadirs_on_the_last[k_dict] + 'conf_mat_tes.csv'
    lpath_cls_report = datadirs_on_the_last[k_dict] + 'cls_report_tes.csv'
    conf_mats_on_the_last.append(pd.read_csv(lpath_conf_mat).rename(columns={"Unnamed: 0": "Mouse#0{}".format(i_mouse+1)}))
    cls_reports_on_the_last.append(pd.read_csv(lpath_cls_report).rename(columns={"Unnamed: 0": "Mouse#0{}".format(i_mouse+1)}))

  conf_mats_on_the_last = pd.concat(conf_mats_on_the_last, axis=1)
  cls_reports_on_the_last = pd.concat(cls_reports_on_the_last, axis=1)
  pprint(conf_mats_on_the_last)
  pprint(cls_reports_on_the_last)


  ## Save
  savedir = '../results/191027/'
  os.makedirs(savedir, exist_ok=True)
  conf_mats_on_the_way.to_csv(savedir + 'conf_mats_on_the_way_5CV.csv')
  cls_reports_on_the_way.to_csv(savedir + 'cls_reports_on_the_way_5CV.csv')
  conf_mats_on_the_last.to_csv(savedir + 'conf_mats_on_the_last_5CV.csv')
  cls_reports_on_the_last.to_csv(savedir + 'cls_reports_on_the_last_5CV.csv')






  '''



































# # ## Prepair
# # df = pd.DataFrame()
# # df['lfp'] = lfp
# # df['rip_start'] = 0
# # rip_sec['start'] = (rip_sec['start_time']*p['samp_rate']).astype(int)
# # df['rip_start'].iloc[rip_sec['start']] = 1

# torch.backends.cudnn.enabled = True

# # Predict Probability density fucntions # fixme
# for i_lfp in range(1, len(lfps)):
#   print('Now predicting from LFP #{}'.format(i_lfp))
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
