import sys
sys.path.append('/mnt/nvme/proj/progs/06_File_IO')
sys.path.append('/mnt/nvme/proj/progs/07_Learning/')
sys.path.append('/mnt/nvme/proj/progs/')
import os

from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
import multiprocessing as mp
from models_pt import EncBiSRU_binary as Model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import time
import utils.myfunc as mf


def train(epoch):
  model.train()
  dl_tra = mf.mk_dataloader(d['fpaths_tra'], d['samp_rate'], d['max_seq_len'], d['use_pertubation_tra'], d['bs_tra'])
  d['n_tra'] = len(dl_tra.dataset)
  d['epoch'].append(epoch)
  train_counter = 0
  for i, batch in enumerate(dl_tra):
    # i, batch = next(enumerate(dl_tra))
    Xb, Tb_dur, Tb_lat = batch
    Xb, Tb_dur, Tb_lat = Xb.to(d['device']), Tb_dur.to(d['device']), Tb_lat.to(d['device'])
    Tb_isn = (Tb_lat < 100 / 1000).to(torch.long) # imbalance, resampling should be one idea
    # d['Tb_lat_tra'].append(Tb_lat.cpu().detach().numpy())

    optimizer.zero_grad()

    pred_dur, pred_lat, pred_prob_isn = model(Xb)
    pred_cls_isn = pred_prob_isn.argmax(dim=1)

    loss_dur = criterion_mse(pred_dur.squeeze(), Tb_dur)
    loss_dur = loss_dur * 1
    loss_dur = loss_dur.mean()

    loss_lat = criterion_mse(pred_lat.squeeze(), Tb_lat)
    loss_lat = loss_lat * 1
    loss_lat = loss_lat.mean()

    loss_isn = criterion_xentropy(pred_prob_isn, Tb_isn)
    weights = mf.calc_weight(Tb_isn, n_class=2)
    loss_isn *= weights
    loss_isn = loss_isn.mean()

    loss_tot = loss_dur + loss_lat + loss_isn

    corrects_isn = (Tb_isn.squeeze() == pred_cls_isn).to(torch.float)
    acc_isn = corrects_isn.mean()

    loss_isn.backward()
    # loss_isn.backward(retain_graph=True)
    # loss_dur.backward(retain_graph=True)
    # loss_lat.backward()
    # loss_tot.backward() # fixme

    optimizer.step()

    train_counter += len(Xb)

    if i % d['log_interval'] == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_isn: {:.4f}\tACC_isn: {:.4f}\tLoss_dur: {:.4f}\tLoss_lat: {:.4f}\tLoss_tot: {:.4f}'\
        .format(epoch, i*d['bs_tra'], d['n_tra'], 100. * i*d['bs_tra'] / d['n_tra'],
                loss_isn, acc_isn, loss_dur, loss_lat, loss_tot))
      d['preds_dur_tra'] = pred_dur.cpu().detach().numpy()
      d['preds_lat_tra'] = pred_lat.cpu().detach().numpy()
      d['tgts_dur_tra'] = Tb_dur.cpu().detach().numpy()
      d['tgts_lat_tra'] = Tb_lat.cpu().detach().numpy()
      d['losses_dur_tra'].append(loss_dur.cpu().item())
      d['losses_lat_tra'].append(loss_lat.cpu().item())
      d['losses_isn_tra'].append(loss_isn.cpu().item())
      d['acc_isn_tra'].append(acc_isn.cpu().item())
      d['losses_tot_tra'].append(loss_tot.cpu().item())
      d['train_counter'].append(train_counter)

  # if epoch % d['save_epoch_interval'] == 0:
  # Save
  savepath_weight = '{}/epoch{}_model.pth'.format(save_dir, epoch)
  torch.save(model.state_dict(), savepath_weight)
  print('Saved to {}'.format(savepath_weight))
  savepath_optim = '{}/epoch{}_optimizer.pth'.format(save_dir, epoch)
  torch.save(optimizer.state_dict(), savepath_optim)
  print('Saved to {}'.format(savepath_optim))
  savepath_savedict = '{}/epoch{}_savedict.pkl'.format(save_dir, epoch)
  mf.pkl_save(d, savepath_savedict)
  print('Saved to {}'.format(savepath_savedict))


def test():
  dl_tes = mf.mk_dataloader(d['fpaths_tes'], d['samp_rate'], d['max_seq_len'], d['use_pertubation_tes'], d['bs_tes'])
  model.eval()
  d['n_tes'] = len(dl_tes.dataset)
  cum_loss_dur = 0
  cum_loss_lat = 0
  cum_loss_isn = 0
  cum_correct_isn = 0
  test_counter = 0
  with torch.no_grad():
    for i, batch in enumerate(dl_tes):
      # i, batch = next(enumerate(dl_tes))
      # Target
      Xb, Tb_dur, Tb_lat = batch
      Xb, Tb_dur, Tb_lat = Xb.to(d['device']), Tb_dur.to(d['device']), Tb_lat.to(d['device'])
      Tb_isn = (Tb_lat < 100 / 1000).to(torch.long)

      pred_dur, pred_lat, pred_prob_isn = model(Xb)
      pred_cls_isn = pred_prob_isn.argmax(dim=1)

      loss_dur = criterion_mse(pred_dur.squeeze(), Tb_dur).sum()

      loss_lat = criterion_mse(pred_lat.squeeze(), Tb_lat).sum()

      # outputs.shape [N, C], labels.shape [N,]
      loss_isn = criterion_xentropy(pred_prob_isn, Tb_isn)
      weights = mf.calc_weight(Tb_isn, n_class=2)
      loss_isn *= weights
      loss_isn = loss_isn.sum()

      corrects_isn = (Tb_isn.squeeze() == pred_cls_isn).to(torch.float)

      cum_loss_dur += loss_dur
      cum_loss_lat += loss_lat
      cum_loss_isn += loss_isn
      cum_correct_isn += corrects_isn.sum()
      test_counter += len(Xb)

      d['preds_dur_tes'] = pred_dur.cpu().numpy()
      d['preds_lat_tes'] = pred_lat.cpu().numpy()
      d['preds_isn_tes'] = pred_prob_isn.cpu().numpy()
      d['tgts_dur_tes'] = Tb_dur.cpu().numpy()
      d['tgts_lat_tes'] = Tb_lat.cpu().numpy()
      d['tgts_isn_tes'] = Tb_isn.cpu().numpy()

  ave_loss_dur = 1.0* cum_loss_dur / test_counter
  ave_loss_lat = 1.0* cum_loss_lat / test_counter
  ave_loss_isn = 1.0* cum_loss_isn / test_counter
  acc_isn = 1.0* cum_correct_isn / test_counter
  ave_loss_tot = ave_loss_dur + ave_loss_lat

  d['losses_dur_tes'].append(ave_loss_dur.cpu().item())
  d['losses_lat_tes'].append(ave_loss_lat.cpu().item())
  d['losses_isn_tes'].append(ave_loss_isn.cpu().item())
  d['acc_isn_tes'].append(acc_isn.cpu().item())
  d['losses_tot_tes'].append(ave_loss_tot.cpu().item())

  print('\nTest set: Avg. Loss_isn: {:.4f}, ACC_isn: {:.4f}, Loss_dur: {:.4f}, Loss_lat: {:.4f}, Loss_tot: {:.4f}\n'\
        .format(ave_loss_isn, acc_isn, ave_loss_dur, ave_loss_lat, ave_loss_tot))

def main():
  for epoch in range(1, d['max_epochs']+1):
    timer('Train epoch {} start'.format(epoch))
    test()
    train(epoch)
    if timer.from_prev_hhmmss:
      d['time_by_epoch'].append(timer.from_prev_hhmmss)


if __name__ == "__main__":
  ## Preparation
  ts = datetime.datetime.fromtimestamp(time.time()).strftime('%y%m%d/%H%M%S')
  print('Time Stamp {}'.format(ts))
  save_dir = '../results/{}'.format(ts)
  os.makedirs(save_dir, exist_ok=True)
  d = defaultdict(list)
  timer = mf.time_tracker()

  ## CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  d['device'] = torch.device("cuda:0" if  use_cuda else "cpu")
  d['n_gpus'] = int(torch.cuda.device_count())
  print('n_gpus : {}'.format(d['n_gpus']))

  ## Load Paths
  loadpath_npy_list = '../data/2kHz_npy_list.pkl'
  d['n_load'] = 176 # 23 # -> fast, 176 -> w/o 05/day5, 186 -> full
  fpaths = mf.pkl_load(loadpath_npy_list)[:d['n_load']]
  d['tes_keyword'] = '02'
  print('Test Keyword: {}'.format(d['tes_keyword']))
  d['fpaths_tra'], d['fpaths_tes'] = mf.split_fpaths(fpaths, tes_keyword=d['tes_keyword'])

  ## Parameters
  d['samp_rate'] = 1000
  d['max_seq_len'] = 1000
  d['use_pertubation_tra'] = True
  d['use_pertubation_tes'] = False
  d['bs_tra'] = 64 * d['n_gpus']
  d['bs_tes'] = 64 * d['n_gpus']
  d['max_epochs'] = 10
  d['lr'] = 1e-3
  d['save_epoch_interval'] = 10
  d['log_interval'] = 1000
  # NN
  d['n_features'] = 1
  d['hidden_size'] = 64
  d['num_layers'] = 4
  d['dropout'] = 0.1
  d['bidirectional'] = True

  ## Initialize Neural Network
  model = Model(input_size=d['n_features'], hidden_size=d['hidden_size'], num_layers=d['num_layers'],
                dropout=d['dropout'], bidirectional=d['bidirectional'])
  model.cuda()
  model = nn.DataParallel(model).cuda()
  optimizer = optim.Adam(model.parameters())
  criterion_mse = nn.MSELoss(reduction='none')
  criterion_xentropy = nn.CrossEntropyLoss(reduction='none') # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

  # Run main loop
  main()


## Evaluating, plot graph
mf.plot_learning_curve(d)
