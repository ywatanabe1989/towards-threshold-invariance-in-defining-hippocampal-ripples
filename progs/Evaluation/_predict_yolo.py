## my own packages
import sys
sys.path.append('./')
# sys.path.append('./utils')
import myutils.myfunc as mf
sys.path.append('./06_File_IO')
from dataloader_yolo_191120 import DataloaderFulfiller
sys.path.append('./07_Learning/')
from optimizers import Ranger
from schedulers import cyclical_lr
from apex import amp
sys.path.append('./10_Evaluation/')
from glob_the_best_model_dir import glob_the_last_model_dir
sys.path.append('./11_Models/')
sys.path.append('./11_Models/yolo')
sys.path.append('./11_Models/yolo/utils')
from yolo.models import Darknet
from yolo.data_parallel import DataParallel
from utils.utils import non_max_suppression_1D as nms
from utils.utils import check_samples_1D, plot_prediction_1D
# ### my own packages
# import sys
# sys.path.append('./')
# import myutils.myfunc as mf
# sys.path.append('./06_File_IO')
# from dataloader_yolo_191028 import DataloaderFulfiller
# sys.path.append('./07_Learning/')
# from balance_xentropy_loss import BalanceCrossEntropyLoss
# from apex import amp
# sys.path.append('./11_Models/yolo')
# # from model_191015 import Model

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


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-nmt", "--n_mouse_tes", default="01", choices=["01", "02", "03", "04", "05"], help = " ")
ap.add_argument("-o", "--use_opti", default='ranger', choices=['ranger', 'sdg'], help = " ")
# 12: Fast, 176: w/o 05/day5, 186: Full
ap.add_argument("-nl", "--n_load_all", default=186, choices=[12, 24, 186], type=int, help = " ")
ap.add_argument("-ca", "--calc_anchor", action='store_true', help = " ")
ap.add_argument("-sch", "--use_scheduler", default="none", nargs='+', choices=["none", "step", "clc"], help = " ")
ap.add_argument("-ilr", "--initial_learning_rate", default=1e-3, type=float, help=" ")
ap.add_argument("-bf", "--batch_factor", default=4, type=int, help=" ")
ap.add_argument("--model_def", default='./11_Models/yolo/config/yolov3.cfg', help=" ")
ap.add_argument("-sr", "--samp_rate", default=1000, type=int, help=" ")
ap.add_argument("-msl", "--max_seq_len", default=416, choices=[52, 104, 384, 416], type=int, help=" ") # input_len
args = ap.parse_args()


def load_params(dirpath_root, plot=False):
  timer = mf.time_tracker()
  # datadir = mf.natsorted_glob(dirpath_root + 'epoch_*/batch_*/')[-1]
  last_dir = mf.natsorted_glob(dirpath_root + 'epoch_*/batch_*/')[-1]
  print('Data directory: {}'.format(last_dir))
  d = mf.pkl_load(last_dir + 'data.pkl')
  p = mf.pkl_load(last_dir + 'params.pkl')
  m = mf.pkl_load(last_dir + 'model.pkl')
  p['bs_tes'] = 1024 * 10
  weight_path = last_dir + 'weight.pth'
  return d, p, m, timer, weight_path, last_dir


def init_the_train_dataloader(d, p, m):
  print('Initializing the train dataloader')
  keys_to_pack_ripple_detect = ['Xb', 'Tb_CenterX_W', 'Tb_levels']

  kwargs_dl = {'samp_rate':p['samp_rate'],
               'use_fp16':m['use_fp16'],
               'use_shuffle':True,
               'max_seq_len_pts':p['max_seq_len'],
               'step':None,
               'use_perturb':True,
               'define_ripples':True,
               'keys_to_pack':keys_to_pack_ripple_detect,
               'bs':p['bs_tra'],
               'nw':20,
               'pm':False, # RuntimeError: Caught RuntimeError in pin memory thread for device 0.
               'drop_last':True,
               }

  dl_fulf_tra = DataloaderFulfiller(p['fpaths_tra'], **kwargs_dl)
  d['n_samples_tra'] = dl_fulf_tra.get_n_samples()
  d['n_batches_per_epoch'] = math.ceil(d['n_samples_tra'] / p['bs_tra'])

  p['save_batch_interval'] = max(1, math.floor(d['n_batches_per_epoch']/p['n_save_per_epoch']))
  p['print_log_batch_interval'] = max(1, math.floor(d['n_batches_per_epoch']/p['n_print_log_per_epoch']))

  return dl_fulf_tra, d, p, m


def init_the_test_dataloader(d, p, m):
  print('Initializing the test dataloader')
  keys_to_pack_ripple_detect = ['Xb', 'Tb_CenterX_W', 'Tb_levels']

  kwargs_dl = {'samp_rate':p['samp_rate'],
               'use_fp16':m['use_fp16'],
               'use_shuffle':True,
               'max_seq_len_pts':p['max_seq_len'],
               'step':None,
               'use_perturb':True,
               'define_ripples':True,
               'keys_to_pack':keys_to_pack_ripple_detect,
               'bs':p['bs_tra'],
               'nw':20,
               'pm':False, # RuntimeError: Caught RuntimeError in pin memory thread for device 0.
               'drop_last':True,
               }

  dl_fulf_tes = DataloaderFulfiller(p['fpaths_tes'], **kwargs_dl)
  return dl_fulf_tes, d, p, m


def init_NN(p, m):
  print('Initializing NN')
  model = Darknet(args.model_def, # fixme
                  dim=1,
                  n_classes=1,
                  input_len=args.max_seq_len,
                  anchors=m['anchors'],
                  ).to(m['device'])

  adam_eps = 1e-8
  optimizer = Ranger(model.parameters(), lr=m['init_lr'], eps=adam_eps)

  if m['n_gpus'] > 1:
    model = DataParallel(model).to(m['device'])

  scheduler = None

  if m['use_scheduler'] == 'step':
    step_all = d['n_batches_per_epoch'] * p['max_epochs']
    last_relat_lr = 0.1
    gamma = np.exp(np.log(last_relat_lr) / step_all)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) # 10.0

  if m['use_scheduler'] == 'clc':
    step_epochs = 2
    step_size = (n_tra / p['bs_tra']) * step_epochs # 100
    clc = cyclical_lr(step_size, min_lr=1.0, max_lr=10.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clc])

  return model, optimizer, scheduler, p, m


def NN_to_eval_mode(model, weight_path):
  ## Load weight
  model_state_dict = torch.load(weight_path)
  model.load_state_dict(model_state_dict)
  model.eval()
  return model


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

        '''
        plt.scatter(Tb_distances_ms.cpu().detach().numpy(),
                    pred_isRipple_probs[:, 1].cpu().detach().numpy())
        plt.xlabel('Distance from Ripple Peak[ms]')
        plt.ylabel('Predicted Ripple Probability (Ripple-center score)')


        n_max_plot = 5
        n_plot = 0
        for _ in range(1000):
          i = np.random.randint(p['bs_tes'])
          distance_ms = Tb_distances_ms[i]
          predicted_prob = pred_isRipple_probs[i, 1].cpu().detach().numpy()
          true_ripple = -p['max_distance_ms'] < distance_ms and distance_ms < p['max_distance_ms']
          pred_ripple = predicted_prob > 0.5
          # if (not true_ripple) and pred_ripple:
          # if true_ripple and pred_ripple:
          if (not true_ripple) and pred_ripple:
            plt.plot(np.arange(-int(len(Xb[0])/2), int(len(Xb[0])/2)), Xb[i].cpu().detach().numpy())
            plt.xlabel('Input Time [msec]')
            plt.ylabel('Amplitude [uV]')
            plt.title('Distance {:.2f} ms, Predicted Probabilitiy {:.2f}'\
                       .format(distance_ms, predicted_prob))
            plt.pause(10)
            plt.close()
            n_plot += 1
            if n_plot == n_max_plot:
              break
        '''

      if m['use_loss'] == 'kldiv':
        Tb_distances_ms_onehot = cvt_Tb_distance_ms(Tb_distances_ms.clone(), p['max_distance_ms'])
        pred_isRipple_logprobs = log_softmax(pred_isRipple_logits)
        pred_isRipple_probs = pred_isRipple_logprobs.exp()
        loss_isRipple = kldiv_criterion(pred_isRipple_logprobs, Tb_distances_ms_onehot.to(m['device'])).sum(dim=-1)

        '''
        n_max_plot = 10
        n_plot = 0
        for _ in range(1000):
          i = np.random.randint(p['bs_tra'])
          distance_ms = Tb_distances_ms[i]
          true_ripple = -p['max_distance_ms'] < distance_ms and distance_ms < p['max_distance_ms']
          if not true_ripple:
            print(distance_ms)
            plt.bar(np.arange(-(p['max_distance_ms']+1), (p['max_distance_ms']+1)+1), pred_isRipple_probs[i].detach().cpu())
            plt.title('Distance {} [ms]'.format(distance_ms))
            plt.pause(4)
            plt.close()
            n_plot += 1
            if n_plot == n_max_plot:
              break
        '''

      # if m['apply_n_labels_balancing']:
      #   loss_isRipple = balance_loss(loss_isRipple, Tb_isRipple) # Scaling wrt sample sizes

      # if m['apply_distance_adjustament']:
      #   lam = get_lambda(abs(Xb_Tbs_dict['Tb_distances_ms']), max_distance_ms=p['max_distance_ms'])
      #   loss_isRipple = (lam.to(torch.float).to(m['device'])*loss_isRipple) # Scaling wrt distances

      # # plt.scatter(Tb_distances_ms.cpu().detach().numpy(), loss_isRipple.cpu().detach().numpy())
      # loss_isRipple = loss_isRipple.mean()

      # if m['use_multitaskloss']:
      #   losses = torch.stack([loss_isRipple])
      #   loss_tot = (multitaskloss(losses)).sum()
      # else:
      #   loss_tot = loss_isRipple

      # cum_dict['cum_loss_isRipple_tes'] += loss_isRipple.cpu().detach().numpy().astype(np.float)
      # cum_dict['cum_loss_tot_tes'] += loss_tot.cpu().detach().numpy().astype(np.float)
      # cum_dict['n_counter_tes'] += len(Xb)

      # print('{:.1f}%'.format(i_batch/n_batches*100))
      if i_batch == n_batches:
        break

  # cum_dict = _calc_ave(cum_dict)
  # _print_log_tes(cum_dict)
  # _save_tes(cum_dict)

  # timer('Epoch {} Test ends'.format(d['epoch'][-1]))

def plot_prediction(dl, conf_thres=0.001, nms_thres=.5, max_plot=5):
  batch = next(iter(dl))
  Xb, targets = batch
  '''
  check_with_plotting(Xb, targets)
  '''
  Xb.cuda()
  # Xb = Xb[:4].cuda()
  # targets = targets[:20]
  outputs = model(Xb)
  outputs = nms(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
  plot_prediction_1D(Xb, targets, outputs, max_plot=max_plot)


dirpath_root = '/mnt/md0/proj/results/191127/201011/'
d, p, m, timer, weight_path, data_dir = load_params(dirpath_root)
dl_fulf_tra, d, p, m = init_the_train_dataloader(d, p, m)
dl_fulf_tes, d, p, m = init_the_test_dataloader(d, p, m)
model, optimizer, scheduler, p, m = init_NN(p, m)
model = NN_to_eval_mode(model, weight_path)

## Plot
dl_tra = dl_fulf_tra()
dl_tes = dl_fulf_tes()

# plot_prediction(dl_tes, conf_thres=0.1, nms_thres=.3)
# plot_prediction(dl_tes, conf_thres=.7, nms_thres=.001, max_plot=10)
# plot_prediction(dl_tes, conf_thres=.5, nms_thres=.1e-3, max_plot=10)
# plot_prediction(dl_tes, conf_thres=1e-3, nms_thres=1e-3, max_plot=10)
plot_prediction(dl_tes, conf_thres=1e-2, nms_thres=1e-2, max_plot=10)
plot_prediction(dl_tes, conf_thres=1e-2, nms_thres=0.3, max_plot=10)


## check outputs, targets with digits
conf_thres, nms_thres = 1e-2, .3
dl_tra = dl_fulf_tra()
batch = next(iter(dl_tra))
Xb, targets = batch
# Xb.cuda()
Xb = Xb[:4].cuda()
targets = targets[:20]
outputs = model(Xb)
# outputs = nms(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
targets[:,2:] *= Xb.shape[-1]
pprint(targets)
pprint(outputs)



if __name__ == '__main__':
  ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  ap.add_argument("-nmt", "--n_mouse_tes", default=1, choices=[1,2,3,4,5], type=int, help = "") #  '191022/230633'
  ap.add_argument("-save", action='store_true', help = "")
  ap.add_argument("-plot", action='store_true', help = "")
  args = ap.parse_args()

  timestamps = {
  'Test Mouse Number 01':'191023/212438',
  'Test Mouse Number 02':'191024/132223',
  'Test Mouse Number 03':'191025/004805',
  'Test Mouse Number 04':'191025/132659',
  'Test Mouse Number 05':'191026/011033',
  }

  load_params(timestamps['Test Mouse Number 0{}'.format(args.n_mouse_tes)], plot=args.plot)

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
