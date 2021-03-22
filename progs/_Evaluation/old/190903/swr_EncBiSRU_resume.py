import sys
sys.path.append('/mnt/nvme/proj/progs/06_File_IO')
sys.path.append('/mnt/nvme/proj/progs/07_Learning/')
sys.path.append('/mnt/nvme/proj/progs/')

from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
import multiprocessing as mp
from models_pt import EncBiSRU
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

    optimizer.zero_grad()
    pred_dur, pred_lat, h_n = model(Xb)

    loss_dur = criterion(pred_dur.squeeze(), Tb_dur)
    loss_dur = loss_dur * 1 # fixme: balancing, perplexity
    loss_dur = loss_dur.mean()

    loss_lat = criterion(pred_lat.squeeze(), Tb_lat)
    loss_lat = loss_lat * 1 # fixme: balancing, perplexity
    loss_lat = loss_lat.mean()

    loss_tot = loss_dur + loss_lat

    loss_dur.backward(retain_graph=True)
    loss_lat.backward()
    # loss_tot.backward() # fixme

    optimizer.step()

    train_counter += len(Xb)

    if i % d['log_interval'] == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_dur: {:.4f}\tLoss_lat: {:.4f}\tLoss_tot: {:.4f}'\
        .format(epoch, i*d['bs_tra'], d['n_tra'], 100. * i*d['bs_tra'] / d['n_tra'], loss_dur, loss_lat, loss_tot))
      d['preds_dur_tra'] = pred_dur.cpu().detach().numpy()
      d['preds_lat_tra'] = pred_lat.cpu().detach().numpy()
      d['tgts_dur_tra'] = Tb_dur.cpu().detach().numpy()
      d['tgts_lat_tra'] = Tb_lat.cpu().detach().numpy()
      d['losses_dur_tra'].append(loss_dur.cpu().item())
      d['losses_lat_tra'].append(loss_lat.cpu().item())
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
  test_counter = 0
  with torch.no_grad():
    for i, batch in enumerate(dl_tes):
      # i, batch = next(enumerate(dl_tes))

      Xb, Tb_dur, Tb_lat = batch
      Xb, Tb_dur, Tb_lat = Xb.to(d['device']), Tb_dur.to(d['device']), Tb_lat.to(d['device'])

      pred_dur, pred_lat, h_n = model(Xb)

      loss_dur = criterion(pred_dur.squeeze(), Tb_dur).sum()
      loss_lat = criterion(pred_lat.squeeze(), Tb_lat).sum()

      cum_loss_dur += loss_dur
      cum_loss_lat += loss_lat
      test_counter += len(Xb)

      d['preds_dur_tes'] = pred_dur.cpu().numpy()
      d['preds_lat_tes'] = pred_lat.cpu().numpy()
      d['tgts_dur_tes'] = Tb_dur.cpu().numpy()
      d['tgts_lat_tes'] = Tb_lat.cpu().numpy()

  ave_loss_dur = cum_loss_dur / test_counter
  ave_loss_lat = cum_loss_lat / test_counter
  ave_loss_tot = ave_loss_dur + ave_loss_lat

  d['losses_dur_tes'].append(ave_loss_dur.cpu().item())
  d['losses_lat_tes'].append(ave_loss_lat.cpu().item())
  d['losses_tot_tes'].append(ave_loss_tot.cpu().item())

  print('\nTest set: Avg. Loss_dur: {:.4f}, Loss_lat: {:.4f}, Loss_tot: {:.4f}\n'\
        .format(ave_loss_dur, ave_loss_lat, ave_loss_tot))

# def main():
#   for epoch in range(1, d['max_epochs']+1):
#     timer('Train epoch {} start'.format(epoch))
#     test()
#     train(epoch)
#     if timer.from_prev_hhmmss:
#       d['time_by_epoch'].append(timer.from_prev_hhmmss)

def resume():
  global model, d, optimizer, criterion
  lpath_model = '../results/{}/epoch{}_model.pth'.format(load_ts, load_epoch)
  lpath_optim = '../results/{}/epoch{}_optimizer.pth'.format(load_ts, load_epoch)
  lpath_savedict = '../results/{}/epoch{}_savedict.pkl'.format(load_ts, load_epoch)

  d = mf.pkl_load(lpath_savedict)

  model = EncBiSRU(input_size=d['n_features'], hidden_size=d['hidden_size'], num_layers=d['num_layers'],
                   dropout=d['dropout'], bidirectional=d['bidirectional'])
  model = model.cuda()
  model = nn.DataParallel(model)
  optimizer = optim.Adam(model.parameters(), lr=d['lr'])

  model_state_dict = torch.load(lpath_model)
  model.load_state_dict(model_state_dict)
  optimizer_state_dict = torch.load(lpath_optim)
  optimizer.load_state_dict(optimizer_state_dict)

  criterion = nn.MSELoss(reduction='none')

  for epoch in range(d['epoch'][-1]+1, d['epoch'][-1] + 10):
    timer = mf.time_tracker()
    timer('Train epoch {} start'.format(epoch))
    test()
    train(epoch)
    if timer.from_prev_hhmmss:
      d['time_by_epoch'].append(timer.from_prev_hhmmss)

if __name__ == "__main__":

  ## Resume Training
  # load_ts = '190719_193109'
  # load_epoch = 10

  # Run main loop
  resume()


## Evaluating, plot graph
mf.plot_learning_curve(d)


# plot()
