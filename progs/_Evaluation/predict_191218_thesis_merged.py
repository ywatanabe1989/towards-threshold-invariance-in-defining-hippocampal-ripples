### my own packages
import sys
sys.path.append('./')
sys.path.append('./myutils')
import myfunc as mf
sys.path.append('./06_File_IO')
# from dataloader_191020 import dataloader_fulfiller
from dataloader_191217_thesis import dataloader_fulfiller
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
from scipy.optimize import curve_fit

'''
python 10_Evaluation/predict_191218_thesis_merged.py -nmt 1;\
python 10_Evaluation/predict_191218_thesis_merged.py -nmt 2;\
python 10_Evaluation/predict_191218_thesis_merged.py -nmt 3;\
python 10_Evaluation/predict_191218_thesis_merged.py -nmt 4;\
python 10_Evaluation/predict_191218_thesis_merged.py -nmt 5;\
'''


## Funcs
def load_params(ldir, plot=False):
  global timer, d, p, m, weight_path, datadir
  timer = mf.time_tracker()
  print('Data directory: {}'.format(ldir))
  d = mf.pkl_load(ldir + 'data.pkl')
  p = mf.pkl_load(ldir + 'params.pkl')
  m = mf.pkl_load(ldir + 'model.pkl')
  # m['use_opti'] = 'sgd' # fixme
  p['bs_tes'] = 10240 # 512 * 1 # fixme, 1024*10 for full
  weight_path = ldir + 'weight.pth'


def init_dataloader(used_EMG):
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

  if not used_EMG:
      label_name = None

  if used_EMG:
      label_name = 'label_cleaned_from_gmm'.format(args.n_mouse_tes)

  kwargs_dl_tes = {'ripples_binary_classification':False,
                   'ripples_multi_classification':True,
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
                   'label_name':label_name,
                   }
  #                 'used_EMG':used_EMG,

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

  if m['use_loss'] == 'xentropy':
    global softmax, xentropy_criterion
    softmax = torch.nn.Softmax(dim=-1)
    xentropy_criterion = torch.nn.CrossEntropyLoss(reduction='none')

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


class Recorder():
    def __init__(self,):
        self.Tb_SD = []
        self.pred_isRipple_prob = []
    def add_SD(self, Tb_SD):
        self.Tb_SD.append(Tb_SD.cpu().numpy())
    def add_prob(self, pred_isRipple_prob):
        self.pred_isRipple_prob.append(pred_isRipple_prob.cpu().numpy())


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
      Xb_Tbs_dict = mf.init_dict(keys=dl_fulf_tes.kwargs['keys_to_pack'], values=batch)

      # to GPU
      for k in dl_fulf_tes.kwargs['keys_to_pack']:
        Xb_Tbs_dict[k] = Xb_Tbs_dict[k].to(m['device'])

      # Model Outputs
      Xb = Xb_Tbs_dict['Xb'] # fixme
      pred_isRipple_logits = model(Xb.unsqueeze(-1))

      if m['use_loss'] == 'xentropy':
        Tb_SD = Xb_Tbs_dict['Tb_SD']
        recorder.add_SD(Tb_SD)
        pred_isRipple_probs = softmax(pred_isRipple_logits)[:,1]
        recorder.add_prob(pred_isRipple_probs)
      if i_batch == n_batches:
        break


def sigmoid(x, a, b):
     y = 1 / (1 + np.exp(-b*(x-a)))
     return y


def to_onehot(a, num_classes=2):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-nmt", "--n_mouse_tes", default=1, choices=[1,2,3,4,5], type=int, help = "") #  '191022/230633'
    ap.add_argument("-save", action='store_true', help = "")
    ap.add_argument("-plot", action='store_true', help = "")
    args = ap.parse_args()


    used_EMG = True

    ## Load
    # if not used_EMG:
    #     timestamps = {
    #     'Test Mouse Number 01':'191219/205715',
    #     'Test Mouse Number 02':'191219/063042',
    #     'Test Mouse Number 03':'191219/064927',
    #     'Test Mouse Number 04':'191219/062310',
    #     'Test Mouse Number 05':'191219/064946',
    #     }


    if used_EMG:
        # timestamps = {
        # 'Test Mouse Number 01':'191219/082538',
        # 'Test Mouse Number 02':'191218/232620',
        # 'Test Mouse Number 03':'191218/232632',
        # 'Test Mouse Number 04':'191218/232641',
        # 'Test Mouse Number 05':'191218/232652',
        # }

        # Half GAP and w/o Normalization on Input
        # timestamps = {
        # 'Test Mouse Number 01':'191226/051124',
        # 'Test Mouse Number 02':'191227/020119',
        # 'Test Mouse Number 03':'191227/111012',
        # 'Test Mouse Number 04':'191226/165627',
        # 'Test Mouse Number 05':'191227/204739',
        # }

        # GAP and Normalization Input
        timestamps = {
        'Test Mouse Number 01':'191228/200745',
        'Test Mouse Number 02':'191228/110403',
        'Test Mouse Number 03':'191228/110415',
        'Test Mouse Number 04':'191228/110429',
        'Test Mouse Number 05':'191228/110439',
        }

    ts = timestamps['Test Mouse Number 0{}'.format(args.n_mouse_tes)]

    dirpath_root = '../results/' + ts + '/'

    dirpaths = mf.natsorted_glob(dirpath_root + 'epoch_*/batch_*/')
    lpath = dirpaths[-1]
    load_params(lpath, plot=args.plot)
    print(p['tes_keyword'])
    assert args.n_mouse_tes == int(p['tes_keyword'])

    init_dataloader(used_EMG)
    init_ripple_detector_NN()
    NN_to_eval_mode()

    ## Collect NN's Outputs
    recorder = Recorder()
    test(n_batches=d['n_batches_per_epoch_tes'])
    # Format
    pred_probs = np.vstack(recorder.pred_isRipple_prob).reshape(-1,1)
    SDs = np.vstack(recorder.Tb_SD).reshape(-1,1)

    np.save(lpath + 'pred_probs_gap_norminp.npy', pred_probs)
    np.save(lpath + 'SDs_gap_norminp.npy', SDs)
