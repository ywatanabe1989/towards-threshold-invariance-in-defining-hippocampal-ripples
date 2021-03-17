from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.signal
from scipy.signal import find_peaks, wiener
from numba import jit
import sys
import math
from multiprocessing import Pool
import multiprocessing as mp
import gc
import torch
# import torchvision
import torch.utils.data as utils
from tqdm import tqdm
import random
from bisect import bisect_left, bisect_right
from kymatio import Scattering1D
import sys
from glob import glob
import cv2
import re
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import pandas as pd
from obspy.signal.tf_misfit import cwt
from obspy.imaging.cm import obspy_sequential
import pandas as pd

# from itertools import chain
# import inspect
# import pandas as pd
# import pickle
# import bz2

################### PRINT ###################
def boxplot(data, labels, sym='', ylim=None, title=None, notch=True):
    ## Boxplot
    medianprops = dict(color='black')
    fig, ax = plt.subplots(1,1)
    ax.boxplot(data, sym=sym, notch=notch, medianprops=medianprops)
    ax.set_xticklabels(labels, rotation=0, fontsize=14)
    ax.set_title(title)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    if ylim is not None:
        ax.set_ylim(ylim)

def plt_line_2D(arr, title=None, xlabel=None, ylabel=None,\
                xmin=None, xmax=None,
                ymin=None, ymax=None,
                paintx=None, painty=None,
                grid=None, wait_sec=None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.plot(range(arr.shape[0]), arr)
    plt.grid(grid)
    if paintx:
        for i,j in paintx:
            ax.axvspan(i, j, color='blue', alpha=0.3)
    if painty:
        for i,j in painty:
            ax.axhspan(i, j, color='yellow', alpha=0.3)
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    fig.show()
    if wait_sec > 0:
        time.sleep(wait_sec)
        plt.close()


def plt_scatter_2D(arr, wait_sec=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
        xlabel = 'array[:,0]'
        ylabel = 'array[:,1]'
    else:
        xlabel = 'array[0]'
        ylabel = 'array[1]'

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(arr[0], arr[1])

    ax.set_title('scatter plot')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.show()
    if wait_sec > 0:
        time.sleep(wait_sec)
        plt.close()

################### ML ###################
def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
                print(name, param.data)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def print_parameters(model):
  for name, param in model.named_parameters():
      if param.requires_grad:
          print(name, param.data)

def mk_MNIST_dataloader(mldata_dir='../../data/mldata', train=True, bs=64, nw=10, pm=True, \
                                               collate_fn=None):
  import torchvision
  mnist = torchvision.datasets.MNIST(mldata_dir, download=True, train=train)
  X, T = mnist.data, mnist.targets
  X = im2col(X).unsqueeze(-1) # 3rd dimension is n_features = 1
  X_1st, X_2nd = X.chunk(2, dim=1)
  dataset = utils.TensorDataset(X_1st, X_2nd, T)
  if collate_fn:
    dataloader = utils.DataLoader(dataset, batch_size=bs, shuffle=train, num_workers=nw, \
                                pin_memory=True, collate_fn=collate_fn)
  else:
    dataloader = utils.DataLoader(dataset, batch_size=bs, shuffle=train, num_workers=nw, \
                                pin_memory=True)
  return dataloader

def ts_extract_features_from_array(array):
  import pandas as pd
  import tsfresh
  df = pd.DataFrame(array.squeeze().cpu.numpy())
  features = tsfresh.extract_features(df)

def get_mnist_sk1(dist_dir):
    from sklearn import datasets
    mnist = datasets.fetch_mldata('MNIST original', data_home=dist_dir)
    X = mnist.data
    Y = mnist.target
    return X, Y

def get_mnist_sk2(n_in=28, n_time=28):
  n_cls = 10

  from sklearn import datasets
  from sklearn.model_selection import train_test_split
  mnist = datasets.fetch_mldata('MNIST original', data_home='.')
  n_all = len(mnist.data)
  n_use = n_all # Use a part of MNIST data set
  indices = np.random.permutation(range(n_all))[:n_use]  # pickup N samples randomly
  p_test = 0.1
  n_test =  (int)(n_use * p_test)
  n_train = n_use - n_test

  X = mnist.data[indices]
  X = 1.0 * X / X.max()
  X = X - X.mean(axis=1)[:,np.newaxis]
  X = X.reshape(len(X), n_time, n_in)
  T = mnist.target[indices]
  # T = np.eye(10)[T.astype(int)] # onehot

  X_tra, X_tes, T_tra, T_tes = train_test_split(X, T, test_size=n_test)
  N_tra = len(X_tra)
  return X_tra, T_tra, X_tes, T_tes


class MyBayes():
  def __init__(self, fx, params_dict, n_linspace=101, n_iter=15):
    import numpy as np
    from gpflowopt.domain import ContinuousParameter
    self.fx = fx
    self.params_dict = params_dict
    self.iter_count = 0
    for key,value in params_dict.iteritems():
      if self.iter_count == 0:
        self.domain = ContinuousParameter(key, value[0], value[1])
      else:
        self.domain += ContinuousParameter(key, value[0], value[1])
      self.iter_count += 1
    self.r = None
    self.n_linspace = n_linspace
    self.n_iter = n_iter
    self.X = None
    self.Y = None

  def gpr(self):
    import gpflow
    from gpflowopt.bo import BayesianOptimizer
    from gpflowopt.design import LatinHyperCube
    from gpflowopt.acquisition import ExpectedImprovement
    from gpflowopt.optim import SciPyOptimizer
    from numba import jit

    # domain = ContinuousParameter('x1', 0, 20) + ContinuousParameter('x2', -1, 2) + ContinuousParameter('x3', -1, 2)

    # Use standard Gaussian process Regression
    n_linspace = self.n_linspace #10001 # like batch_size
    lhd = LatinHyperCube(n_linspace, self.domain)

    self.X = lhd.generate()
    self.Y = self.fx(self.X)

    model = gpflow.gpr.GPR(self.X, self.Y, gpflow.kernels.Matern52(2, ARD=True)) # ???
    model.kern.lengthscales.transform = gpflow.transforms.Log1pe(1e-3) # ???

    # Now create the Bayesian Optimizer
    alpha = ExpectedImprovement(model)
    optimizer = BayesianOptimizer(self.domain, alpha)

    # Run the Bayesian optimization
    with optimizer.silent():
        self.r = optimizer.optimize(self.fx, n_iter=self.n_iter) # minimize fx output
    print(self.r)

def split_train_and_test_data(lfp, rip_sec, fpaths, tes_keyword=None):
  all_indices = list(range(len(lfp)))
  tes_indices = [i for i, f in enumerate(fpaths) if f.find(tes_keyword) > 0]
  tra_indices = get_train_indices(all_indices, tes_indices) # mf.
  # lfp_tra = lfp[tra_indices] # duplicated on memory, del lfp_tra,mf.slice_by_index(lfp, tra_indices)
  # rip_sec_tra = rip_sec[tra_indices] # mf.slice_by_index(rip_sec, tra_indices)
  # lfp_tes = lfp[tes_indices] # mf.slice_by_index(lfp, tes_indices)
  # rip_sec_tes = rip_sec[tes_indices]  # mf.slice_by_index(rip_sec, tes_indices)

  lfp_tra = slice_by_index(lfp, tra_indices)
  rip_sec_tra = slice_by_index(rip_sec, tra_indices)
  lfp_tes = slice_by_index(lfp, tes_indices)
  rip_sec_tes = slice_by_index(rip_sec, tes_indices)
  return lfp_tra, rip_sec_tra, lfp_tes, rip_sec_tes

def get_train_indices(all_indices, test_indices):
  train_indices = []
  for i in all_indices:
    if not i in test_indices:
      train_indices.append(i)
  assert len(all_indices) == len(test_indices) + len(train_indices)
  return train_indices


def batch2inputs_tra(batch, max_seq_len, samp_rate): # after getting batch from dataloader
  Xb = batch[0]
  Tb = batch[1]
  # print('Xb.shape : {}'.format(Xb.shape)) # [8, 91710581]
  # print('Tb.shape : {}'.format(Tb.shape)) # [8, 4, 44738]
  # print('Xb.dtype : {}'.format(Xb.dtype)) # float32
  # print('Tb.dtype : {}'.format(Tb.dtype)) # float32

  Xb_sliced = []
  Tb_dur = []
  Tb_lat = []
  # for i in tqdm(range(len(Xb))):
  for i in range(len(Xb)):
    Xi = Xb[i]
    Xi_length = count_non_pad(Xi)
    Ti = Tb[i]
    set_flag = False
    while not set_flag:
      try:
        slice_start = random.randint(0, Xi_length - max_seq_len) # min: 0, max: Xi_length - max_seq_len
        # slice_start = len(Xi) - max_seq_len - 750
        slice_end = slice_start + max_seq_len
        slice_end_sec = slice_end / samp_rate

        Xi_sliced = Xi[slice_start:slice_end]

        after_slice_end_mask = slice_end_sec < Ti[0]
        next_ripple_ind = after_slice_end_mask.nonzero()[0] # index 0 is out of bounds for dimension 0 with size 0

        next_ripple_dur = Ti[2][next_ripple_ind] # check this is correct
        # next_ripple_lat = Ti[3][next_ripple_ind]
        from_slice_end_to_the_next_ripple_start = Ti[0][next_ripple_ind] - slice_end_sec

        Xb_sliced.append(Xi_sliced)
        Tb_dur.append(next_ripple_dur)
        Tb_lat.append(from_slice_end_to_the_next_ripple_start)
        set_flag = True
      except:
        pass
  Xb_sliced = torch.stack(Xb_sliced, dim=0)
  Tb_dur = torch.stack(Tb_dur, dim=0)
  Tb_lat = torch.stack(Tb_lat, dim=0)
  return Xb_sliced, Tb_dur, Tb_lat

class MyCollator(object):
  def __init__(self, *params):#max_seq_len=1000):
    self.params = params
  def __call__(self, batch):
    max_seq_len = self.params[0]
    samp_rate = self.params[1]

    Xb = []
    Tb = []
    for sample in batch:
        X, T = sample
        Xb.append(X)
        Tb.append(T)
    Xb = torch.stack(Xb, dim=0)
    Tb = torch.stack(Tb, dim=0)
    # print('Xb.shape : {}'.format(Xb.shape)) # [8, 91710581]
    # print('Tb.shape : {}'.format(Tb.shape)) # [8, 4, 44738]
    # print('Xb.dtype : {}'.format(Xb.dtype)) # float16
    # print('Tb.dtype : {}'.format(Tb.dtype)) # float16

    # Slice LFP and get next ripple param
    Xb = Xb.to(torch.float32)
    Tb = Tb.to(torch.float32)
    batch = (Xb, Tb)
    # print('max_seq_len {}'.format(max_seq_len)) # 1000
    # print('samp_rate {}'.format(samp_rate)) # 1000
    Xb_sliced, Tb_dur, Tb_lat = batch2inputs_tra(batch, max_seq_len, samp_rate)

    return Xb_sliced, Tb_dur, Tb_lat

## Padding ##
def padding_from_numpylist(numpylist):
  for i in range(len(numpylist)):
    numpylist[i] = torch.tensor(numpylist[i], dtype=torch.float16)
  return torch.nn.utils.rnn.pad_sequence(numpylist, batch_first=True, padding_value=-float("Inf"))

def count_non_pad(tensor1d, pad=-float('inf')):
  pad_mask = (tensor1d == pad)
  num_pad = pad_mask.sum().data
  length = -num_pad + len(tensor1d)
  return int(length)

def balance_loss(loss, Tb, counts_arr=None, n_classes_int=None):
  if not 'torch' in dir():
    import torch

  # define n_classes_int
  if not n_classes_int:
    try:
      n_classes_int = len(counts_arr)
    except:
      n_classes_int = int(Tb.max())+1

  # define counts_arr
  try:
    _ = counts_arr.shape
    counts_arr = torch.FloatTensor(counts_arr)
  except:
    counts_arr = torch.zeros(n_classes_int)
    for i in range(n_classes_int):
      counts_arr[i] += (Tb == i).sum()

  weights = torch.zeros_like(Tb, dtype=torch.float)
  probs_arr = 1. * counts_arr / counts_arr.sum()
  non_zero_mask = (probs_arr > 0)
  recip_probs_arr = torch.zeros_like(probs_arr)
  recip_probs_arr[non_zero_mask] = probs_arr[non_zero_mask] ** (-1)
  for i in range(n_classes_int):
    mask = (Tb == i)
    weights[mask] += recip_probs_arr[i]
  weights_norm = (weights / weights.mean()).to(loss.dtype).to(loss.device)
  loss *= weights_norm
  return loss

'''
if __name__ == '__main__':
  ## TEST ##
  def test(loss, Tb, counts_arr=None, n_classes_int=None, title=None):
    loss_orig = loss.clone()
    balanced_loss = balance_loss(loss.clone(), Tb, counts_arr=counts_arr, n_classes_int=n_classes_int)
    print()
    print(title)
    print('batched targets: {}'.format(Tb))
    print('loss_orig: {}'.format(loss_orig))
    print('counts_arr: {}'.format(counts_arr))
    print('n_classes_int: {}'.format(n_classes_int))
    print('balanced_loss: {}'.format(balanced_loss))
    if loss.mean() == balanced_loss.mean():
      print('balanced_loss.mean() is the same as loss.mean()')
    print()

  loss = torch.Tensor([1,1,1,1,1,1])
  Tb = torch.LongTensor([0,0,0,1,1,2])

  test(loss, Tb, counts_arr=None,                n_classes_int=None, title='w/o counts nor n_classes_int')
  test(loss, Tb, counts_arr=np.array([1, 2, 3]), n_classes_int=None, title='w/ counts, w/o n_classes_int')
  test(loss, Tb, counts_arr=None,                n_classes_int=4,    title='w/ n_classes_int, w/o  counts_arr')
  test(loss, Tb, counts_arr=np.array([1,2,3,4]), n_classes_int=4,    title='w/ both n_classes_int and counts_arr')
  ##########
'''

def get_pred_wf(_pred_wf, _longests, pred_dur, samp_rate):
  longests = _longests.cpu().numpy()
  u, ind = np.unique(longests, return_index=True)
  longests = u[np.argsort(ind)].astype(np.int)
  __pred_wf = []
  start = 0
  for i in range(len(longests)):
    end = start+longests[i]
    __pred_wf.append(_pred_wf[start:end].transpose(0,1))
    start = end
  pred_wf = []
  for i in range(len(__pred_wf)):
    __pred_wf_tmp = __pred_wf[i]
    for j in range(len(__pred_wf_tmp)):
        pred_wf.append(__pred_wf_tmp[j])
  for i in range(len(pred_wf)):
      pred_wf[i] = (pred_wf[i][:int(pred_dur[i]*samp_rate)])
  return pred_wf, longests


################### IO ###################
def csv_read(fname, skiprows=3):
    import numpy as np
    import pandas as pd
    import dask.dataframe as ddf
    import dask.multiprocessing
    import gc

    df = ddf.read_csv(fname, skiprows=skiprows)
    df = df.compute(get=dask.multiprocessing.get)

    arr = np.array(df, dtype=np.float64).T
    del df
    return arr

def save_npy(np_arr, fpath):
    np.save(fpath, np_arr)
    print('Saved to: {}'.format(fpath))

def pkl_save(obj, fpath):
    import pickle
    with open(fpath, 'wb') as f: # 'w'
        pickle.dump(obj, f)
    print('Saved to: {}'.format(fpath))

def save_pkl(obj, fpath):
    pkl_save(obj, fpath)

def dill_save(obj, fpath):
    import dill
    with open(fpath, 'wb') as f: # 'w'
        dill.dump(obj, f)
    print('Saved to: {}'.format(fpath))

def hdf5_save(obj_list, name_list, fpath):
    import h5py
    with h5py.File(fpath, 'w') as hf:
      for (name, obj) in zip(name_list, obj_list):
        hf.create_dataset(name,  data=obj)
    # hf = h5py.File(fpath, 'wb')
    # hf.create_dataset(name, data=obj)
    # hf.close()
    print('Saved to: {}'.format(fpath))

def hdf5_load(name_list, fpath):
    import h5py
    data = {}
    with h5py.File(fpath, 'r') as hf:
      for name in name_list:
        data_tmp = hf[name][:]
        data[name] = data_tmp
    return data
        # hf = h5py.File(fpath, 'rb')
        # hf.keys()
        # n1 = hf.get(name)



def pkl_load(fpath):
    import pickle
    with open(fpath, 'rb') as f: # 'r'
        obj = pickle.load(f)
        # print(obj.keys())
        return obj

def load_pkl(fpath):
    obj = pkl_load(fpath)
    return obj

def dill_load(fpath):
    import dill
    with open(fpath, 'rb') as f: # 'r'
        obj = dill.load(f)
        # print(obj.keys())
        return obj

def pkl2_save(obj, fpath):
    import pickle
    with open(fpath, 'wb') as f: # 'w'
        pickle.dump(obj, f, protocol=4)
    print('Saved to: {}'.format(fpath))


def pkl2_load(fpath):
    import pickle
    with open(fpath, 'rb') as f: # 'r'
        obj = pickle.load(f)
        # print(obj.keys())
        return obj

def pkl4_save(obj, fpath):
    import pickle
    with open(fpath, 'wb') as f: # 'w'
        pickle.dump(obj, f, protocol=4)
    print('Saved to: {}'.format(fpath))


def pkl4_load(fpath):
    import pickle
    with open(fpath, 'rb') as f: # 'r'
        obj = pickle.load(f)
        # print(obj.keys())
        return obj

class ZpkObj:
    '''
    obj = ZpkObj(obj) # compress
    obj = obj.load()

    zobj1 = ZpkObj(obj1) #
    del obj1
    '''
    import pickle
    import bz2
    PROTOCOL = pickle.HIGHEST_PROTOCOL
    def __init__(self, obj):
        self.zpk_object = bz2.compress(pickle.dumps(obj, PROTOCOL), 9)
    def load(self):
        return pickle.loads(bz2.decompress(self.zpk_object))


################### DSP ###################
def quasi_attractor_transformation(data, tau=3): # data [bs, seq_len, n_features=1]
    qa = []
    for i in range(tau):
      qa.append(data[:, i:-tau+i])
    qa = torch.stack(qa).transpose(0,-1).squeeze()
    return qa

class bandpass():
  def __init__(self, lowcut=100, highcut=250, fs=1000):
    self.lowcut = lowcut
    self.highcut = highcut
    self.fs = fs

  def mk_butter_bandpass(self, order=5):
    from scipy.signal import butter, sosfilt, sosfreqz
    nyq = 0.5 * self.fs
    low = self.lowcut / nyq
    high = self.highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

  def butter_bandpass_filter(self, data):
    from scipy.signal import butter, sosfilt, sosfreqz
    sos = self.mk_butter_bandpass()
    y = sosfilt(sos, data)
    return y

def bandpass(data, lo_hz, hi_hz, fs, order=5):
  def mk_butter_bandpass(order=5):
    from scipy.signal import butter, sosfilt, sosfreqz
    nyq = 0.5 * fs
    low = lo_hz / nyq
    high = hi_hz / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

  def butter_bandpass_filter(data):
    from scipy.signal import butter, sosfilt, sosfreqz
    sos = mk_butter_bandpass()
    y = sosfilt(sos, data)
    return y

  sos = mk_butter_bandpass()
  return


# def wavelet(signal, delta_sec, out_path=None, y_max=256):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from wavelets import WaveletAnalysis

#     x = signal # given a signal x(t)
#     dt = delta_sec  # and a sample spacing
#     wa = WaveletAnalysis(x, dt=dt)
#     power = wa.wavelet_power # wavelet power spectrum
#     scales = wa.scales # scales
#     t = wa.time    # associated time vector
#     rx = wa.reconstruction() # reconstruction of the original data

#     fig, ax = plt.subplots()
#     T, S = np.meshgrid(t, scales)
#     ax.contourf(T, S, power, 100)
#     ax.set_yscale('log')
#     ax.set_ylim([0, y_max])
#     ax.set_ylabel('Power [Hz]')
#     ax.set_xlabel('Time [{} * {} sec]'.format(len(t), dt**(-1)))
#     fig.show()
#     return wa, power, scales, t, rx
#     if out_path:
#         fig.savefig(out_path)

def wavelet_scattering_1D(x, J=8, Q=12):
  T = x.shape[1]
  # J = 8 # the maximum scale of the scattering transform is set to `2**J = 2**8 = 256
  # Q = 12 # wavelet per octave
  scattering = Scattering1D(J, T, Q).cuda()
  Sx = scattering.forward(x.squeeze().to(torch.float)) # J=8 => [256, 337, 3], J=9 => [256, 435, 1]
  Sx = Sx[:,1:,:]
  log_eps = 1e-6
  Sx = torch.log(torch.abs(Sx) + log_eps)
  Sx = torch.mean(Sx, dim=-1) # J=8 => [256, 336], J=9 => [256, 434],
  return Sx

def fft_torch(Xb):
  Xb_ = Xb.clone()
  zeros = torch.zeros_like(Xb_)
  # sp = torchaudio.transforms.Spectrogram(n_fft=1000, power=1)
  # spectrums = sp(Xb_[0].to(torch.float).cpu().transpose(0,1))
  Xb_ = torch.cat((Xb_, zeros), dim=-1).to(torch.float)
  length = Xb.shape[1]
  fft_out = Xb_.fft(signal_ndim=1, normalized=True)
  power = 2.0
  powers = torch.norm(fft_out, 2, -1).pow(power)[:, :int(length/2)]
  # powers[:] = powers[:] / powers.sum(axis=1, keepdim=True)
  # powers[:] = (powers[:] - powers.mean(axis=1, keepdim=True)) / powers.std(axis=1, keepdim=True)
  powers = (powers - powers.mean(axis=0, keepdim=True)) / powers.std(axis=0, keepdim=True)
  return powers

# def cwt(sig):
#     import pywt
#     import numpy as np
#     import matplotlib.pyplot as plt

#     widths = np.arange(1, 900001)
#     return pywt.cwt(sig, widths, 'mexh')

#     plt.imshow(cwtmatr, extent=[0, 90001, 1, 51], cmap='PRGn', \
#                aspect='auto', vmax=abs(cwtmatr).max(), \
#                vmin=-abs(cwtmatr).max())
#                         # doctest: +SKIP
#     plt.show() # doctest: +SKIP

def ssinwave(A, omega, t, k, x, sigma):
    y = A * np.sin( omega * t - k * x + sigma)
    return y

def randomize(arg):
    randomized_arg = np.random.randn() + arg
    return randomized_arg
    #test = np.array([randomize(100) for i in xrange(1000)]) #confirm

def mk_a_morlet(A, M, w, s):
    A = randomize(A)
    M = randomize(M).astype(np.int)
    M = len(M)
    w = randomize(w)
    s = randomize(s)
    mlt = scipy.signal.morlet(M, w, s)
    mlt = A * mlt
    return A, M, w, s, mlt

def chkprint(*args):
    from inspect import currentframe
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    print(', '.join(names.get(id(arg),'???')+' = '+repr(arg) for arg in args))

def init_sess():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess

def train_test_split(X, Y):
    from sklearn.model_selection import train_test_split
    X_train,  X_test, Y_train, Y_test = train_test_split(X, Y)
    print('X_train, X_test, Y_train, Y_test')
    return X_train, X_test, Y_train, Y_test

################### General ###################
'''
def myprint(args, fname='out.txt'):
  sys.stdout =open(fname, 'a')
  print(args)
  sys.stdout = sys.__stdout__
  print(args)
'''

def rgb2bgr(rgb_colors):
    return (rgb_colors[1],
            rgb_colors[0],
            rgb_colors[2])

def get_color_code(color_name, use_rgb=True):
    RGB_PALLETE = {
        'blue':(0,128,192),
        'red':(255,70,50),
        'pink':(255,150,200),
        'green':(20,180,20),
        'yellow':(230,160,20),
        'glay':(128,128,128),
        'parple':(200,50,255),
        'light_blue':(20,200,200),
        'blown':(128,0,0),
        'navy':(0,0,100),
    }

    if use_rgb == True: # RGB
        color_code = RGB_PALLETE[color_name]
    else: # BGR
        color_code = rgb2bgr(RGB_PALLETE[color_name])

    return color_code

def get_plot_color(color_name):
    color_code = get_color_code(color_name, use_rgb=True)
    return np.array(color_code) / 255


def calc_partial_corrcoef(x, y, z):
    '''remove the influence of the variable z from the correlation between x and y.'''
    r_xy = np.corrcoef(x, y)
    r_xz = np.corrcoef(x, z)
    r_yz = np.corrcoef(y, z)
    r_xy_z = (r_xy - r_xz*r_yz) / (1-r_xz**2)*(1-r_yz**2)
    return r_xy_z

def term_plot(y_arr, rows=10, columns=200):
       import terminalplot as tplt
       x_arr = np.arange(len(y_arr))
       tplt.plot(list(x_arr), list(y_arr), rows=rows, columns=columns)


def get_peaks(arr_1d, N=30, prominence=50, distance=60, do_plot=False):
    # wiener filtering
    filted = arr_1d
    for i in tqdm(range(N)):
      filted = wiener(filted, mysize=9)

    peaks, _ = find_peaks(filted, prominence=prominence, distance=distance)
    print('Unique peak intervals: {}'.format(np.unique(np.diff(peaks))))

    if do_plot:
      plt.plot(arr_1d, label='arr_1d')
      plt.plot(filted, label='filted')
      plt.scatter(peaks, filted[peaks], laebl='peaks')
    '''
    # Check with random indexing
    start = np.random.randint(len(arr_1d)) - 1000
    end = start + 1000
    x = np.arange(start, end)
    plt.plot(x, arr_1d[start:end], label='arr_1d')
    plt.plot(x, filted[start:end], color='blue', label='filtered')
    _peaks = peaks[(start < peaks ) * (peaks < end)]
    plt.scatter(_peaks, filted[_peaks], color='red', label='peaks')
    plt.legend()
    n_frames = len(peaks)
    fps = 15
    print('Video Time %.3f [hour]' %(1.* n_frames / fps / 3600))
    '''

    '''
    # Check if false peaks were detected
    idx = onsets[int(np.where(np.diff(peaks) == 107)[0])]
    start = idx - 1000
    end = idx + 1000
    plt.plot(arr_1d[start:end], label='arr_1d')
    plt.plot(filted[start:end], label='filtered')
    plt.legend()
    _peaks = peaks[(start < peaks) * (peaks < end)]
    plt.scatter(_peaks, resid[_peaks], label='peaks')
    '''
    return peaks



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


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def search_str_list(str_list, search_key):
  matched_keys = []
  for string in str_list:
    m = re.search(search_key, string)
    if m is not None:
      matched_keys.append(string)
  return matched_keys

# def shuffle_dict(dict):
#   '''
#   dict contents should be np.array
#   '''
#   keys = dict.keys()
#   shuffled_values = shuffle(*dict.values())
#   for i, k in enumerate(keys):
#     dict[k] = shuffled_values[i]
#   return dict

def shuffle_dict(dict):
  keys = list(dict.keys())
  values = list(dict.values())
  lengths = np.array([len(values[i]) for i in range(len(values))])
  assert (lengths == lengths[0]).all()
  # rand_indi = torch.randperm(int(lengths[0]))
  rand_indi = np.random.permutation(int(lengths[0]))
  rand_values = [v[rand_indi] for v in values]
  for i, k in enumerate(keys):
    dict[k] = rand_values[i]
  return dict


def listed_dict(keys=None):
  from collections import defaultdict
  dict_list = defaultdict(list)
  # initialize with keys if possible
  if keys is not None:
    for k in keys:
      dict_list[k] = []
  return dict_list

def init_dict(keys=None, values=None):
  dict = {}

  if values is None:
    values = [0 for _ in range(len(keys))]

  if keys is not None:
    for i, k in enumerate(keys):
      dict[k] = values[i]

  return dict


def take_closest(list_obj, num_insert):
    """
    Assumes list_obj is sorted. Returns closest value to num.
    If two numbers are equally close, return the smallest number.
    list_obj = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    num = 3.5
    mf.take_closest(list_obj, num)
    # output example (3, 3)
    """
    if math.isnan(num_insert):
      return np.nan, np.nan

    pos_num_insert = bisect_left(list_obj, num_insert)

    if pos_num_insert == 0: # When the insertion is at the first position
        closest_num = list_obj[0]
        closest_pos = pos_num_insert # 0
        return closest_num, closest_pos

    if pos_num_insert == len(list_obj): # When the insertion is at the last position
        closest_num = list_obj[-1]
        closest_pos = pos_num_insert # len(list_obj)
        return closest_num, closest_pos

    else: # When the insertion is anywhere between the first and the last positions
      pos_before = pos_num_insert - 1

      before_num = list_obj[pos_before]
      after_num = list_obj[pos_num_insert]

      delta_after = abs(after_num - num_insert)
      delta_before = abs(before_num - num_insert)

      if delta_after < delta_before:
         closest_num = after_num
         closest_pos = pos_num_insert

      else: # if delta_before <= delta_after:
         closest_num = before_num
         closest_pos = pos_before

      return closest_num, closest_pos


def sorted_glob(text):
  return sorted(glob(text))

def natsorted_glob(text):
  import natsort
  return natsort.natsorted(glob(text))


class file_existence_checker():
  def __init__(self):
    import os
    self.os = os
    self.exist = 0
    self.nonexist = 0

  def check_a_file(self, fname):
    exist = self.os.path.isfile(fname)
    if exist:
      # print('{} exists.'.format(fname))
      self.exist += 1
    if not exist:
      print('{} doesn\'t exist.'.format(fname))
      self.nonexist += 1

  def check_files(self, fnames_list):
    for f in fnames_list:
      self.check_a_file(f)
    print('{}/{} files exists'.format(self.exist, self.exist + self.nonexist))


def pdf(x, mu, sigma): # probability density function
  var = sigma**2
  return 1./((2*math.pi*var)**0.5) * torch.exp(-(x-mu)**2 / (2*var))

def pad_npy_list(npy_list, longest=None):
  npy_lengths = [len(npy_list[i]) for i in range(len(npy_list))]
  if longest is None:
    longest = max(npy_lengths) # 91710581
  ## Padding
  for j in range(len(npy_list)):
    pad_len = longest-len(npy_list[j])
    npy_list[j] = np.pad(npy_list[j].squeeze(), (0, pad_len) )
  npy_list = np.stack(npy_list)
  return npy_list, npy_lengths

## DEBUGGING ##

def findlocals(search, startframe=None, trace=False):

    from pprint import pprint
    import inspect, pdb

    startframe = startframe or sys.last_traceback
    frames = inspect.getinnerframes(startframe)

    frame = [tb for (tb, _, lineno, fname, _, _) in frames
             if search in (lineno, fname)][0]

    if trace:
        pprint(frame.f_locals)
        pdb.set_trace(frame)
    return frame.f_locals

def get_func_def(func):
    import inspect
    source = inspect.getsource(func)
    print(source)
    # return str(source).decode('euc_jp')

def reload(module):
  import importlib
  module = importlib.reload(module)
  return module

def pdb():
    import pdb; set_trace()
    pass

def split_seq(iterable, size):
    import itertools
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))

@jit
def numericalSort(value):
    import re
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

class time_tracker():
    def __init__(self):
        self.id = -1
        self.start = time.time()
        self.prev = self.start

    def __call__(self, comment=None):
        now = time.time()
        from_start = now - self.start
        self.from_start_hhmmss = time.strftime('%H:%M:%S', time.gmtime(from_start))
        from_prev = now - self.prev
        self.from_prev_hhmmss = time.strftime('%H:%M:%S', time.gmtime(from_prev))
        self.id += 1
        self.prev = now
        if comment:
            print("Time (id:{}): tot {}, prev {} [hh:mm:ss]: {}\n".format(\
                  self.id, self.from_start_hhmmss, self.from_prev_hhmmss, comment))
        else:
            print("Time (id:{}): tot {}, prev {} [hh:mm:ss]\n".format(\
                  self.id, self.from_start_hhmmss, self.from_prev_hhmmss))
@jit
def rotate_list(list_obj, n_rotate):
    if n_rotate == 0:
        return list_obj
    for i in xrange(n_rotate):
        rotated = list_obj[1:] + list_obj[:1]
        list_obj = rotated
    return rotated

class update_mean_var_std_arr():
  def __init__(self, axis=None):
    self.n_prev = 0
    self.mean_prev = np.zeros(1)
    self.var_prev = np.zeros(1)
    self.axis = axis

  def update(self, samp_new):
    samp_new = samp_new.astype(np.float128)
    if self.axis == None:
      if np.all(self.mean_prev) == 0.:
        self.mean_prev = np.mean(samp_new)
      n_new = samp_new.size
      mean_samp = np.mean(samp_new)
      var_samp = np.var(samp_new)
    if not self.axis == None:
      if np.all(self.mean_prev) == 0.:
        self.mean_prev = np.mean(samp_new, axis=self.axis)
      n_new = samp_new.shape[self.axis]
      mean_samp = np.mean(samp_new, axis=self.axis)
      var_samp = np.var(samp_new, axis=self.axis)
    mean_new = 1.0*(mean_samp*n_new + self.mean_prev*self.n_prev)/(n_new+self.n_prev)

    var_new = 1.0 * (self.n_prev*self.var_prev + self.n_prev*((mean_new-self.mean_prev)**2) \
                  +         n_new*var_samp     +       n_new*((mean_new-mean_samp)**2)) \
                  / (self.n_prev+n_new)
    std = var_new**(0.5)

    self.n_prev += n_new
    self.mean_prev = mean_new
    self.var_prev = var_new
    return mean_new, var_new, std

def split_fpath(fpath):
    import os
    dirname = os.path.dirname(fpath) + '/'
    base = os.path.basename(fpath)
    fname, ext = os.path.splitext(base)
    return dirname, fname, ext

class file_existence_checker():
  def __init__(self):
    import os
    self.os = os
    self.exist = 0
    self.nonexist = 0

  def check_a_file(self, fname):
    exist = self.os.path.isfile(fname)
    if exist:
      # print('{} exists.'.format(fname))
      self.exist += 1
    if not exist:
      print('{} doesn\'t exist.'.format(fname))
      self.nonexist += 1

  def check_files(self, fnames_list):
    for f in fnames_list:
      self.check_a_file(f)
    print('{}/{} files exists'.format(self.exist, self.exist + self.nonexist))

# def get_size(obj, seen=None):
#     """Recursively finds size of objects"""
#     size = sys.getsizeof(obj)
#     if seen is None:
#         seen = set()
#     obj_id = id(obj)
#     if obj_id in seen:
#         return 0
#     # Important mark as seen *before* entering recursion to gracefully handle
#     # self-referential objects
#     seen.add(obj_id)
#     if isinstance(obj, dict):
#         size += sum([get_size(v, seen) for v in obj.values()])
#         size += sum([get_size(k, seen) for k in obj.keys()])
#     elif hasattr(obj, '__dict__'):
#         size += get_size(obj.__dict__, seen)
#     elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
#         size += sum([get_size(i, seen) for i in obj])
#     return size

# def getsize(obj):
#     import sys
#     from types import ModuleType, FunctionType
#     from gc import get_referents

#     # Custom objects know their class.
#     # Function objects seem to know way too much, including modules.
#     # Exclude modules as well.
#     BLACKLIST = type, ModuleType, FunctionType
#     """sum size of object & members."""
#     if isinstance(obj, BLACKLIST):
#         raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
#     seen_ids = set()
#     size = 0
#     objects = [obj]
#     while objects:
#         need_referents = []
#         for obj in objects:
#             if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
#                 seen_ids.add(id(obj))
#                 size += sys.getsizeof(obj)
#                 need_referents.append(obj)
#         objects = get_referents(*need_referents)
#     return size

def sizeof_fmt(num, suffix='B'): # format string syntax
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def aiueo(obj):
    size = get_size(obj)
    size_h = sizeof_fmt(size)
    return size_h

def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0
    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))
        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))
        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}
        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())
    return sz

def show_objects_size(threshold, unit=2): # how to use?
    disp_unit = {0: 'bites', 1: 'KB', 2: 'MB', 3: 'GB'}
    globals_copy = globals().copy()
    for object_name in globals_copy.keys():
        size = compute_object_size(eval(object_name))
        if size > threshold:
            print('{:<15}{:.3f} {}'.format(object_name, size, disp_unit[unit]))

def compute_object_size(o, handlers={}):
    from collections import deque
    from itertools import chain
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
# show_objects_size(0.1, unit=3)

def retrieve_name_as_list(var):
    import inspect
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names_list = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    return names_list

# class mydict(dict):
#   def __init__(self,):
#     super()

#   def __call__(self, var):
#     callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#     names_list = [var_name for var_name, var_val in callers_local_vars if var_val is var]
#     name = names_list[0]
#     self[name].append(var)
#     self.update({name:var})

class mydict():
  def __init__(self):
    from collections import defaultdict
    self.dict = defaultdict(list)
    import inspect

  def __call__(self, var):

    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names_list = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    name = names_list[0]
    self.dict[name].append(var)

  def appends(self, save_list):
    for var in save_list:
      callers_local_vars = inspect.currentframe().f_back.f_locals.items()
      names_list = [var_name for var_name, var_val in callers_local_vars if var_val is var]
      name = names_list[0]
      self.dict[name].append(var)


def plt_hist(data, bins=100, title=None, xlabel=None, ylabel=None):
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)

  n, bins, patches = ax.hist(data, bins=bins, density=True, stacked=True)
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  plt.show() # fig.show()
  probs = n * np.diff(bins)
  return probs, bins, patches

def slice_by_index(lst, indexes):
    """Slice list by positional indexes.
    Adapted from https://stackoverflow.com/a/9108109/304209.
    Args:
        lst: list to slice.
        indexes: iterable of 0-based indexes of the list positions to return.

    Returns:
        a new list containing elements of lst on positions specified by indexes.
    """
    from operator import itemgetter
    if not lst or not indexes:
        return []
    slice_ = itemgetter(*indexes)(lst)
    if len(indexes) == 1:
        return [slice_]
    return list(slice_)

################### Array ###################
def twoD2oneD(array):
  if array.ndim != 2:
    print('ERROR: Input array is not 2D.')
  else:
    if array.shape[0] == 1:
      array = array[0,:]
      return array
    if array.shape[1] == 1:
      array = array[:,0]
      return array

def display_npy(fname): # fixme
    data = np.load(fname)
    data = twoD2oneD(data)
    mf.plt_line_2D(data[:int(2e5)], title=fname)

def imshow(arr, time_msec=1000):
  import cv2
  import numpy as np
  arr = arr.astype(np.uint8)
  cv2.imshow('', arr)
  cv2.waitKey(time_msec)
  cv2.destroyAllWindows()

def im2col(tensor):
  return tensor.view(len(tensor), -1)

def col2im(tensor, row):
  return tensor.view(len(tensor), row, -1)

def batches(X, Y, batch_size, i=0):
  permutation = torch.randperm(X.size()[0])
  indices = permutation[i:i+batch_size]
  batch_X, batch_Y = X[indices], Y[indices]
  return batch_X, batch_Y

def onehot(tensor, nb_digits=10):
  tensor = tensor.unsqueeze(-1)
  batch_size = tensor.size(0)
  y_onehot = torch.FloatTensor(batch_size, nb_digits)
  y_onehot.zero_()
  y_onehot.scatter_(1, tensor, 1)
  return y_onehot

# class recorder():
#   def __init__(self):
#     self.losses_c_tra = []
#     self.train_counter = []
#     self.losses_c_tes = []
#     self.test_counter = []
#     self.pred_classes = []
#     self.time_by_epoch_str = []
#     self.acc_tes = []
#     self.lr_find_loss = []
#     self.lr_find_lr = []

#   def str2sec(self, time_str):
#       h, m, s = time_str.split(':')
#       return int(h) * 3600 + int(m) * 60 + int(s)

#   def get_mean_sec(self):
#     sec = 0
#     count = 0
#     for str in self.time_by_epoch_str:
#       count += 1
#       sec += str2sec(str)
#     return sec/count


# def get_lfp_and_sampling_rate(lpath):
#   ## Load LFP and Ripple Times, and get sampling rate
#   if lpath.find('2kHz') >= 0:
#     samp_rate = 2000
#     lpath = lpath.replace('2kHz', '1kHz')
#     lpath_rt = lpath.replace('.npy', '_riptimes.pkl')
#   elif lpath.find('1kHz') >= 0:
#     samp_rate = 1000
#     # lpath = lpath.replace('1kHz', '1kHz')
#     lpath_rt = lpath.replace('.npy', '_riptimes.pkl')
#   elif lpath.find('500Hz') >= 0:
#     samp_rate = 500
#     lpath = lpath.replace('500Hz', '1kHz')
#     lpath_rt = lpath.replace('.npy', '_riptimes.pkl')
#   lfp = np.load(lpath).squeeze()
#   rip_sec = pkl_load(lpath_rt)
#   return lfp, rip_sec, samp_rate

# def get_lfp_rip_sec_samprate(lpath):
#   ## Load LFP and Ripple Times, and get sampling rate
#   if lpath.find('2kHz') >= 0:
#     samp_rate = 2000
#     lpath = lpath.replace('2kHz', '1kHz')
#   elif lpath.find('1kHz') >= 0:
#     samp_rate = 1000
#   elif lpath.find('500Hz') >= 0:
#     samp_rate = 500
#     lpath = lpath.replace('500Hz', '1kHz')
#   lpath_rip = lpath.replace('.npy', '_rip_sec.pkl')
#   lfp = np.load(lpath).squeeze()
#   rip_sec = pkl_load(lpath_rip)
#   return lfp, rip_sec, samp_rate

## Computer Vision ##
def find_brightest_spot(image, radius, imshow=False):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (radius, radius), 0)
  (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
  center = maxLoc
  if imshow:
    cv2.circle(image, maxLoc, radius, (255, 0, 0), 2)
    cv2.imshow("found brigtest spot", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  return center

def find_brightest_spot_from_gray(gray, radius, imshow=False):
  # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (radius, radius), 0)
  (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
  center = maxLoc
  if imshow:
    cv2.circle(image, maxLoc, radius, (255, 0, 0), 2)
    cv2.imshow("found brigtest spot", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  return center


################### Current Project Specific ###################
## DSP ##
@jit
def OkadaFilter(Data, alpha=10**2):
    # import numpy as np
    T = len(Data)
    for t in xrange(1,T-1):
        A = Data[t] - Data[t-1]
        B = Data[t] - Data[t+1]
        Data[t] = Data[t] - (A + B)*1.0 / (2*(1 + np.exp(-alpha*A*B)))
    return Data


def get_samp_rate_from_str(str):
  if str.find('2kHz') >= 0:
    samp_rate = 2000
  if str.find('1kHz') >= 0:
    samp_rate = 1000
  if str.find('500Hz') >= 0:
    samp_rate = 500
  return samp_rate

def load_lfp_rip_sec_samprate(lpath_lfp):
  samp_rate = get_samp_rate_from_str(lpath_lfp)
  lfp = np.load(lpath_lfp).squeeze().astype(np.float32) # 2kHz -> int16, 1kHz, 500Hz -> float32

  if samp_rate == 2000:
    lpath_rip = lpath_lfp.replace('2kHz', '1kHz').replace('.npy', '_rip_sec.pkl')
  if samp_rate == 1000:
    lpath_rip = lpath_lfp.replace('1kHz', '1kHz').replace('.npy', '_rip_sec.pkl')
    pass
  if samp_rate == 500:
    lpath_rip = lpath_lfp.replace('500Hz', '1kHz').replace('.npy', '_rip_sec.pkl')

  rip_sec = pkl_load(lpath_rip) # Pandas.DataFrame
  return lfp, rip_sec, samp_rate

def load_lfps_and_rips_pad(fpaths_2kHz, samp_rate=1000): # input: npy, pkl (pands.DataFrame) -> output: torch.Tensor
  lfp = []
  rip_sec = []

  for i in range(len(fpaths_2kHz)):
      lpath_lfp_2kHz = fpaths_2kHz[i]

      if samp_rate == 2000:
        lpath_lfp = lpath_lfp_2kHz
      if samp_rate == 1000:
        lpath_lfp = lpath_lfp_2kHz.replace('2kHz', '1kHz')
      if samp_rate == 500:
        lpath_lfp = lpath_lfp_2kHz.replace('2kHz', '500Hz')

      lfp_tmp, rip_sec_tmp, samp_rate = load_lfp_rip_sec_samprate(lpath_lfp)
      print('Loaded : {}'.format(lpath_lfp))
      lfp_tmp = torch.tensor(lfp_tmp, dtype=torch.float32)
      rip_sec_tmp = torch.tensor(np.array(rip_sec_tmp), dtype=torch.float32) # float16 makes inf
      lfp.append(lfp_tmp)
      rip_sec.append(rip_sec_tmp)
  lfp = torch.nn.utils.rnn.pad_sequence(lfp, batch_first=True, padding_value=-float("Inf"))
  rip_sec = torch.nn.utils.rnn.pad_sequence(rip_sec, batch_first=True, padding_value=-float("Inf"))
  rip_sec = rip_sec.transpose(1,2)
  return lfp, rip_sec, samp_rate

def calc_some_metrices_from_lfp(lfp, rip_sec, samp_rate):
  lfp_sec = 1.* len(lfp) / samp_rate
  print('LFP : {:.1f} hours'.format(lfp_sec / 3600))

  n_rip = len(rip_sec)
  print('The total number of Ripples: {}'.format(n_rip))

  rip_eve_hz = n_rip / lfp_sec
  print('Event frequency: {:.2f} Hz'.format(rip_eve_hz))

  try:
    dur_sec = rip_sec['end_time'] - rip_sec['start_time']
  except:
    dur_sec = rip_sec['end_sec'] - rip_sec['start_sec']
  rip_sec['duration'] = dur_sec
  print('Ave. duration: {:.2f} sec'.format(dur_sec.mean()))

  try:
    latency_sec = rip_sec['start_time'] - rip_sec['end_time'].shift(periods=1)
  except:
    latency_sec = rip_sec['start_sec'] - rip_sec['end_sec'].shift(periods=1)
  rip_sec['latency'] = latency_sec
  print('Ave. latency: {:.2f} sec'.format(latency_sec.mean()))

  return n_rip, rip_eve_hz, rip_sec

def plt_ripple(data, ripple_sec, samp_rate, start_sec=0, end_sec=10, title=None, xlabel=None, ylabel=None):
  time_plt = 1.0*np.arange(start_sec, end_sec, 1.0/samp_rate)
  data_plt = data[start_sec*samp_rate:end_sec*samp_rate]

  try:
    ripple_sec_plt = ripple_sec[start_sec < ripple_sec['start_time']]
    ripple_sec_plt = ripple_sec_plt[ripple_sec_plt['end_time'] < end_sec]
  except:
    ripple_sec_plt = ripple_sec[start_sec < ripple_sec['start_sec']]
    ripple_sec_plt = ripple_sec_plt[ripple_sec_plt['end_sec'] < end_sec]

  f, ax = plt.subplots()

  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  ax.plot(time_plt, data_plt)

  for ripple in ripple_sec_plt.itertuples():
      try:
        ax.axvspan(ripple.start_time, ripple.end_time, alpha=0.3, color='red', zorder=1000)
      except:
        ax.axvspan(ripple.start_sec, ripple.end_sec, alpha=0.3, color='red', zorder=1000)

  plt.show()

def check_ripple(lfp, rip_sec, samp_rate, start_sec=0, end_sec=100, lowcut=100, highcut=250):
  n_rip, rip_eve_hz, rip_sec = calc_some_metrices_from_lfp(lfp, rip_sec, samp_rate)
  # Raw
  plt_ripple(lfp, rip_sec, samp_rate, start_sec=start_sec, end_sec=end_sec, \
             title='Raw', xlabel='Time [s]')
  # Band passed
  bp = bandpass(lowcut=lowcut, highcut=highcut, fs=samp_rate)
  passed = bp.butter_bandpass_filter(lfp)
  plt_ripple(passed, rip_sec, samp_rate, start_sec=start_sec, end_sec=end_sec, \
             title='{}-{} Hz bandpassed'.format(lowcut, highcut), xlabel='Time [s]')

# def calc_some_metrices_from_lfp(lfp, rip_sec, samp_rate):
#   lfp_sec = 1.* len(lfp) / samp_rate
#   print('LFP : {:.1f} hours'.format(lfp_sec / 3600))

#   n_rip = len(rip_sec)
#   print('The total number of Ripples: {}'.format(n_rip))

#   rip_eve_hz = n_rip / lfp_sec
#   print('Event frequency: {:.2f} Hz'.format(rip_eve_hz))

#   dur_sec = rip_sec['end_time'] - rip_sec['start_time']
#   rip_sec['duration'] = dur_sec
#   print('Ave. duration: {:.2f} sec'.format(dur_sec.mean()))

#   latency_sec = rip_sec['start_time'] - rip_sec['end_time'].shift(periods=1)
#   rip_sec['latency'] = latency_sec
#   print('Ave. latency: {:.2f} sec'.format(latency_sec.mean()))

#   # central time
#   # central_times_sec = rip_sec['start_time'] + dur_sec/2.0
#   # rip_sec['central_times'] = central_times_sec
#   # display(rip_sec[:5])
#   return n_rip, rip_eve_hz, rip_sec

# def plt_ripple(data, ripple_sec, samp_rate, start_sec=0, end_sec=10, title=None, xlabel=None, ylabel=None):
#   time_plt = 1.0*np.arange(start_sec, end_sec, 1.0/samp_rate)
#   data_plt = data[start_sec*samp_rate:end_sec*samp_rate]
#   ripple_sec_plt = ripple_sec[start_sec < ripple_sec['start_time']]
#   ripple_sec_plt = ripple_sec_plt[ripple_sec_plt['end_time'] < end_sec]
#   f, ax = plt.subplots()

#   ax.set_title(title)
#   ax.set_xlabel(xlabel)
#   ax.set_ylabel(ylabel)

#   ax.plot(time_plt, data_plt)

#   for ripple in ripple_sec_plt.itertuples():
#       ax.axvspan(ripple.start_time, ripple.end_time, alpha=0.3, color='red', zorder=1000)
#       # ax.axvline(x=ripple.central_times, alpha=0.3, color='green', zorder=1000)
#   plt.show()

# def check_ripple(lfp, rip_sec, samp_rate, start_sec=0, end_sec=100, lowcut=100, highcut=250):
#   n_rip, rip_eve_hz, rip_sec = calc_some_metrices_from_lfp(lfp, rip_sec, samp_rate)
#   # Raw
#   plt_ripple(lfp, rip_sec, samp_rate, start_sec=start_sec, end_sec=end_sec, \
#                 title='Raw', xlabel='Time [sec]')
#   # Band passed
#   bp = bandpass(lowcut=lowcut, highcut=highcut, fs=samp_rate)
#   passed = bp.butter_bandpass_filter(lfp)
#   plt_ripple(passed, rip_sec, samp_rate, start_sec=start_sec, end_sec=end_sec, \
#                 title='{}-{} Hz bandpassed'.format(lowcut, highcut), xlabel='Time [sec]')

def mk_ripple_dataloader(lfp, rip_sec, train=True, bs=64, nw=10, pm=True, collate_fn_class=None, max_seq_len=1000, samp_rate=1000):
  X, T = lfp, rip_sec
  dataset = utils.TensorDataset(X, T) # X and T must be tensor
  if collate_fn_class:
    # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/2
    collate_fn = collate_fn_class(max_seq_len, samp_rate)
    dataloader = utils.DataLoader(dataset, batch_size=bs, shuffle=train, num_workers=nw, \
                                pin_memory=True, collate_fn=collate_fn) # partial(collate_fn_tra, max_seq_len=max_seq_len
  else:
    dataloader = utils.DataLoader(dataset, batch_size=bs, shuffle=train, num_workers=nw, \
                                pin_memory=True)
  return dataloader

def create_inputs_and_targets(lfp, rip_sec, max_seq_len):
  Xb = []
  Tb_dur = []
  Tb_lat = []
  # for i in tqdm(range(len(Xb))):
  for i in range(len(lfp)):
    reshaped_lfp, dur, lat = reshape_an_lfp_and_get_targets(lfp[i], rip_sec[i], max_seq_len)
    Xb.append(reshaped_lfp)
    Tb_dur.append(dur)
    Tb_lat.append(lat)
  # Xb = torch.stack(Xb, dim=0)
  # Tb_dur = torch.stack(Tb_dur, dim=0)
  # Tb_lat = torch.stack(Tb_lat, dim=0)
  return Xb, Tb_dur, Tb_lat


def mold_lfps_and_rips_sec(lfp, rip_sec, samp_rate, max_seq_len):
  lengths = []
  last_rip_starts = []
  for i in range(len(lfp)):
    length = len(lfp[i])
    lengths.append(length)
    a_rip_sec = rip_sec[i]
    last_rip_start_sec = a_rip_sec.loc[len(a_rip_sec)]['start_time']
    last_rip_start = int(last_rip_start_sec * samp_rate)
    last_rip_starts.append(last_rip_start)
  lengths = np.array(lengths)
  last_rip_starts = np.array(last_rip_starts)
  # culc
  last_cut_lengths = lengths - last_rip_starts
  lengths_aligned_by_last_rip_starts = lengths - last_cut_lengths
  shortest_length_aligned_by_last_rip_starts = lengths_aligned_by_last_rip_starts.min()
  first_cut_lengths = lengths_aligned_by_last_rip_starts - shortest_length_aligned_by_last_rip_starts + max_seq_len #
  first_cut_lengths_sec = (first_cut_lengths/samp_rate)
  quotinent = int(shortest_length_aligned_by_last_rip_starts / max_seq_len) - 1
  reminder = shortest_length_aligned_by_last_rip_starts % max_seq_len
  reminder_sec = reminder / samp_rate
  for i in range(len(lfp)):
    lfp[i] = lfp[i][first_cut_lengths[i] + reminder:lengths_aligned_by_last_rip_starts[i]]
    # assert len(lfp[i]) == shortest_length_aligned_by_last_rip_starts
    assert len(lfp[i]) == max_seq_len * quotinent
    rip_sec[i]['start_time'] -= (first_cut_lengths_sec[i] + reminder_sec)
    rip_sec[i]['end_time'] -= (first_cut_lengths_sec[i] + reminder_sec)
    rip_sec[i] = rip_sec[i][0 < rip_sec[i]['start_time']]
  lfp = np.stack(lfp)
  return lfp, rip_sec, samp_rate

def get_targets_from_reshaped_lfp_wrapper(args):
  return get_targets_from_reshaped_lfp(*args)

def multi_get_targets_from_reshaped_lfp(arg_list):
  p = mp.Pool(mp.cpu_count())
  output = p.map(wrapper, arg_list)
  p.close()
  return output

################### Multi Processing Example ###################
'''
def func(n, argument1, argument2):
    return n * argument1 + argument2

def wrapper(args):
    return func(*args)

def multi_process(sampleList):
    p = Pool(8)
    output = p.map(wrapper, sampleList)
    p.close()
    return output

sampleList = [(i, 2, 5) for i in range(num)]
output = multi_process(sampleList)
'''

# class mymultiprocess(): # fixme
#   def __init__(self, func):
#     import multiprocessing as mp
#     self.mp = mp
#     self.func = func

#   def wrapper(self, args):
#       return self.func(*args)

#   def __call__(self, sampleList):
#       p = self.mp.Pool(self.mp.cpu_count())
#       output = p.map(self.wrapper, sampleList)
#       p.close()
#       return output

# def func(self, n, argument1, argument2):
#     return n * argument1 + argument2

# multi = mymultiprocess(func)
# sampleList = [(i, 2, 5) for i in range(100)]
# output = multi(sampleList)

def reshape_an_lfp_and_get_targets(an_lfp, a_rip_sec, max_seq_len): # you can use this for train data with pool, map multiprocessing
  reshaped_lfp = []
  target_dur = []
  target_from_slice_end_to_the_next_ripple_start = []
  an_lfp_length = count_non_pad(an_lfp)
  start = random.randint(0, max_seq_len)
  for i in range(int(an_lfp_length / max_seq_len)):
    try:
      slice_start = start + max_seq_len*i
      slice_end = slice_start + max_seq_len
      slice_end_sec = slice_end / samp_rate

      sliced = an_lfp[slice_start:slice_end] # fixme, reshape or view will be faster

      after_slice_end_mask = slice_end_sec < a_rip_sec[0]
      next_ripple_ind = after_slice_end_mask.nonzero()[0] # index 0 is out of bounds for dimension 0 with size 0
      next_ripple_dur = a_rip_sec[2][next_ripple_ind]
      from_slice_end_to_the_next_ripple_start = a_rip_sec[0][next_ripple_ind] - slice_end_sec
      reshaped_lfp.append(sliced)
      target_dur.append(next_ripple_dur)
      target_from_slice_end_to_the_next_ripple_start.append(from_slice_end_to_the_next_ripple_start)
    except:
      pass
  reshaped_lfp = torch.stack(reshaped_lfp, dim=0)
  target_dur = torch.stack(target_dur, dim=0)
  target_from_slice_end_to_the_next_ripple_start = torch.stack(target_from_slice_end_to_the_next_ripple_start, dim=0)
  return reshaped_lfp, target_dur, target_from_slice_end_to_the_next_ripple_start # 20 secs

def reshape_an_lfp_and_get_targets_wrapper(args): # you can use this for train data with pool, map multiprocessing
    return reshape_an_lfp_and_get_targets(*args)

def multi_reshape_lfp_and_get_targets(arg_list):
    p = mp.Pool(mp.cpu_count())
    output = p.map(reshape_an_lfp_and_get_targets_wrapper, arg_list)
    p.close()
    return output


def get_targets_from_reshaped_lfp(reshaped_lfp, rip_sec, molded_lfp_len, samp_rate, max_seq_len): # 50 min with 1 thread CPU
  target_dur = []
  target_lat = []
  for _i in range(len(reshaped_lfp)):
    # molded_lfps[i, k] == reshaped_lfps[i*quotinent + int(k/max_seq_len), k%max_seq_len]
    # reshaped_lfps[i ,k] == molded_lfps[int(i*max_seq_len/molded_lfp_len), (i*max_seq_len) % molded_lfp_len + k]
    rip_sec_ind = int(_i*max_seq_len/molded_lfp_len)
    slice_end = (_i*max_seq_len) % molded_lfp_len + max_seq_len
    slice_end_sec = slice_end / samp_rate

    next_ripple_idx = bisect_left(rip_sec[rip_sec_ind]['start_time'].values, slice_end_sec)
    next_ripple_dur = rip_sec[rip_sec_ind].iloc[next_ripple_idx]['duration']
    from_slice_end_to_the_next_ripple_start = rip_sec[rip_sec_ind].iloc[next_ripple_idx]['start_time'] - slice_end_sec

    target_dur.append(next_ripple_dur)
    target_lat.append(from_slice_end_to_the_next_ripple_start)
  target_dur = np.stack(target_dur)
  target_lat = np.stack(target_lat)
  return target_dur, target_lat



def load_lfps_and_rips(fpaths_2kHz, samp_rate=1000): # input: npy, pkl (pands.DataFrame) -> output: torch.Tensor
  # print('Loading LFPs and Ripple Times...')
  lfps = []
  rips_sec = []
  for i in range(len(fpaths_2kHz)):
      lpath_lfp_2kHz = fpaths_2kHz[i]

      if samp_rate == 2000:
        lpath_lfp = lpath_lfp_2kHz
      if samp_rate == 1000:
        lpath_lfp = lpath_lfp_2kHz.replace('2kHz', '1kHz')
      if samp_rate == 500:
        lpath_lfp = lpath_lfp_2kHz.replace('2kHz', '500Hz')

      lfp, rip_sec, samp_rate = load_lfp_rip_sec_samprate(lpath_lfp)
      # print('Loaded : {}'.format(lpath_lfp))
      lfps.append(lfp)
      rips_sec.append(rip_sec)
  # print('Loaded.')
  return lfps, rips_sec, samp_rate

def mk_dataset_from_an_lfp_and_a_rip(an_lfp, _a_rip_sec, samp_rate, max_seq_len, perturbation):
  a_rip_sec = _a_rip_sec.copy()
  assert perturbation <= max_seq_len
  last_rip_start = int(a_rip_sec.iloc[-1]['start_time'] * samp_rate)
  quotinent = int(last_rip_start / max_seq_len)
  reminder = last_rip_start % max_seq_len
  assert reminder + max_seq_len*quotinent == last_rip_start
  reference = reminder + max_seq_len
  start = reference - perturbation
  start_sec = start / samp_rate
  # align by start
  an_lfp = an_lfp[start:]
  a_rip_sec['start_time'] -= start_sec
  a_rip_sec['end_time'] -= start_sec
  a_rip_sec = a_rip_sec[0 < a_rip_sec['start_time']]

  Xb = []
  Tb_dur = []
  Tb_lat = []

  for i in range(quotinent-1):
    slice_start = i * max_seq_len
    slice_end = slice_start + max_seq_len
    slice_end_sec = slice_end / samp_rate
    lfp_sliced = an_lfp[slice_start:slice_end]
    higher_idx = bisect_left(a_rip_sec['start_time'].values, slice_end_sec)
    next_rip_dur = a_rip_sec['duration'].iloc[higher_idx]
    next_rip_lat = a_rip_sec['start_time'].iloc[higher_idx] - slice_end_sec
    Xb.append(lfp_sliced)
    Tb_dur.append(next_rip_dur)
    Tb_lat.append(next_rip_lat)
  Xb = np.vstack(Xb)
  Tb_dur = np.hstack(Tb_dur)
  Tb_lat = np.hstack(Tb_lat)
  return Xb, Tb_dur, Tb_lat

def mk_dataset_from_an_lfp_and_a_rip_wrapper(args):
  return mk_dataset_from_an_lfp_and_a_rip(*args)

def multi_mk_dataset_from_lfps_and_rips(arg_list):
    n_cpus = mp.cpu_count()
    p = mp.Pool(n_cpus)
    # print('multiprocessing start ({} cpus)'.format(n_cpus))
    output = p.map(mk_dataset_from_an_lfp_and_a_rip_wrapper, arg_list)
    p.close()
    # print('multiprocessing end')
    return output

def mold_to_dataset(lfps, rips_sec, samp_rate, max_seq_len, include_perturbation):
  if include_perturbation:
    perturbation = [random.randint(0, max_seq_len) for i in range(len(lfps))]
  else:
    perturbation = [0 for i in range(len(lfps))]
  arg_list = [(lfps[i], rips_sec[i], samp_rate, max_seq_len, perturbation[i]) for i in range(len(lfps))]

  output = multi_mk_dataset_from_lfps_and_rips(arg_list) # parallel

  Xb = []
  Tb_dur = []
  Tb_lat = []
  for i in range(len(output)):
    Xb_tmp, Tb_dur_tmp, Tb_lat_tmp = output[i]
    Xb.append(Xb_tmp)
    Tb_dur.append(Tb_dur_tmp)
    Tb_lat.append(Tb_lat_tmp)

  # print('vstack start')
  Xb = np.vstack(Xb) # fixme, make lower memory usage
  # print('vstack end')
  Tb_dur = np.hstack(Tb_dur)
  Tb_lat = np.hstack(Tb_lat)
  return Xb, Tb_dur, Tb_lat

# def mold_to_dataset(lfps, rips_sec, samp_rate, max_seq_len, include_perturbation):
#   if include_perturbation:
#     perturbation = [random.randint(0, max_seq_len) for i in range(len(lfps))]
#   else:
#     perturbation = [0 for i in range(len(lfps))]
#   arg_list = [(lfps[i], rips_sec[i], samp_rate, max_seq_len, perturbation[i]) for i in range(len(lfps))]
#   del lfps, rips_sec
#   gc.collect()

#   arg_list1 = arg_list[int(len(arg_list)/2):]
#   output1 = multi_mk_dataset_from_lfps_and_rips(arg_list1) # parallel
#   Xb_1, Tb_dur_1, Tb_lat_1 = stack(output1)
#   del arg_list1

#   arg_list2 = arg_list[:int(len(arg_list)/2)]
#   del arg_list
#   output2 = multi_mk_dataset_from_lfps_and_rips(arg_list2) # parallel
#   Xb_2, Tb_dur_2, Tb_lat_2 = stack(output2)

#   Xb = np.concatenate((Xb_1, Xb_2), axis=0)
#   Tb_dur = np.concatenate((Tb_dur_1, Tb_dur_2), axis=0)
#   Tb_lat = np.concatenate((Tb_lat_1, Tb_lat_2), axis=0)
#   del Xb_1, Xb_2, Tb_dur_1, Tb_dur_2, Tb_lat_1, Tb_lat2
#   gc.collect()
#   return Xb, Tb_dur, Tb_lat

def stack(output):
  Xb = []
  Tb_dur = []
  Tb_lat = []
  for i in range(len(output)):
    Xb_tmp, Tb_dur_tmp, Tb_lat_tmp = output[i]
    Xb.append(Xb_tmp)
    Tb_dur.append(Tb_dur_tmp)
    Tb_lat.append(Tb_lat_tmp)
  del output, Xb_tmp, Tb_dur_tmp, Tb_lat_tmp
  gc.collect()
  # print('vstack start')
  Xb = np.vstack(Xb) # fixme, make lower memory usage
  # print('vstack end')
  Tb_dur = np.hstack(Tb_dur)
  Tb_lat = np.hstack(Tb_lat)
  return Xb, Tb_dur, Tb_lat

def pack_to_dataloader(X, T1, T2, train=True, bs=64, nw=10, pm=True, collate_fn_class=None, \
                       max_seq_len=1000, samp_rate=1000, use_fp16=False):
  if use_fp16:
    X = torch.FloatTensor(X).to(torch.float16).unsqueeze(-1)
    T1 = torch.FloatTensor(T1).to(torch.float16)
    T2 = torch.FloatTensor(T2).to(torch.float16)
  else:
    X = torch.FloatTensor(X).unsqueeze(-1)
    T1 = torch.FloatTensor(T1)
    T2 = torch.FloatTensor(T2)
  dataset = utils.TensorDataset(X, T1, T2) # X and T must be tensor
  if collate_fn_class:
    # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/2
    collate_fn = collate_fn_class(max_seq_len, samp_rate)
    dataloader = utils.DataLoader(dataset, batch_size=bs, shuffle=train, num_workers=nw, \
                                  pin_memory=True, collate_fn=collate_fn, drop_last=True)
  else:
    dataloader = utils.DataLoader(dataset, batch_size=bs, shuffle=train, num_workers=nw, \
                                  pin_memory=True, drop_last=True)
  return dataloader

# def mk_dataloader(fpaths, samp_rate, max_seq_len, use_pertubation_tra, bs_tra, use_pertubation_tes, bs_tes, tes_keyword):
#   # lfps: list, len(lfps) n_load ->
#   # molded_lfps [n_load, molded_lfp_len] ->
#   # reshaped_lfps [n_load*(molded_lfp_len/max_seq_len), max_seq_len]
#   lfps, rips_sec, samp_rate = load_lfps_and_rips(fpaths, samp_rate=samp_rate) # There is another version using padding.
#   ## Train-Test splitting
#   lfps_tra, rips_sec_tra, lfps_tes, rips_sec_tes = split_train_and_test_data(lfps, rips_sec, fpaths, tes_keyword=tes_keyword)
#   del lfps, rips_sec
#   gc.collect()
#   # mytime(comment='Loaded and Split')
#   Xb_tra, Tb_dur_tra, Tb_lat_tra = mold_to_dataset(lfps_tra, rips_sec_tra, samp_rate, max_seq_len, use_pertubation_tra)
#   del lfps_tra, rips_sec_tra
#   gc.collect()
#   Xb_tes, Tb_dur_tes, Tb_lat_tes = mold_to_dataset(lfps_tes, rips_sec_tes, samp_rate, max_seq_len, use_pertubation_tes)
#   del lfps_tes, rips_sec_tes
#   gc.collect()
#   # mytime(comment='Molded')
#   dataloader_tra = pack_to_dataloader(Xb_tra, Tb_dur_tra, Tb_lat_tra, train=True, bs=bs_tra, \
#                                       collate_fn_class=None, max_seq_len=max_seq_len, samp_rate=samp_rate)
#   del Xb_tra, Tb_dur_tra, Tb_lat_tra
#   gc.collect()
#   dataloader_tes = pack_to_dataloader(Xb_tes, Tb_dur_tes, Tb_lat_tes, train=False, bs=bs_tes, \
#                                            collate_fn_class=None, max_seq_len=max_seq_len, samp_rate=samp_rate)
#   del Xb_tes, Tb_dur_tes, Tb_lat_tes
#   gc.collect()
#   return dataloader_tra, dataloader_tes

def mk_dataloader(fpaths, samp_rate, max_seq_len, use_pertubation, bs, use_fp16, train): # fixed for each of Train dl or Test dl
  # lfps: list, len(lfps) n_load ->
  # molded_lfps [n_load, molded_lfp_len] ->
  # reshaped_lfps [n_load*(molded_lfp_len/max_seq_len), max_seq_len]
  lfps, rips_sec, samp_rate = load_lfps_and_rips(fpaths, samp_rate=samp_rate) # There is another version using padding.
  Xb, Tb_dur, Tb_lat = mold_to_dataset(lfps, rips_sec, samp_rate, max_seq_len, use_pertubation)
  del lfps, rips_sec
  gc.collect()
  if use_fp16:
    Xb, Tb_dur, Tb_lat = Xb.astype(np.float16), Tb_dur.astype(np.float16), Tb_lat.astype(np.float16)
  dataloader = pack_to_dataloader(Xb, Tb_dur, Tb_lat, train=train, bs=bs, \
                                  collate_fn_class=None, max_seq_len=max_seq_len, samp_rate=samp_rate, use_fp16=use_fp16)
  del Xb, Tb_dur, Tb_lat
  gc.collect()
  return dataloader


# def collate_fn_tra(batch, max_seq_len):
#   Xb, Tb = batch
#   Xb = Xb.to(torch.float32)
#   Tb = Tb.to(torch.float32)
#   Tb = Tb.transpose(1,2)

#   Xb_sliced = []
#   Tb_dur = []
#   Tb_lat = []
#   for i in range(len(Xb)):
#     Xi = Xb[i]
#     Xi_length = count_non_pad(Xi)
#     Ti = Tb[i]
#     set_flag = False
#     while not set_flag:
#       try:
#         slice_start = random.randint(0, Xi_length-1 - max_seq_len)
#         slice_end = slice_start + max_seq_len
#         slice_end_sec = slice_end / samp_rate

#         Xi_sliced = Xi[slice_start:slice_end]
#         after_slice_end_mask = Ti[0] > slice_end_sec
#         next_ripple_dur = Ti[2][after_slice_end_mask][0]
#         next_ripple_lat = Ti[3][after_slice_end_mask][0]

#         # if Xi_sliced
#         # if next_ripple_dur
#         # if next_ripple_lat

#         Xb_sliced.append(Xi_sliced)
#         Tb_dur.append(next_ripple_dur)
#         Tb_lat.append(next_ripple_lat)
#         set_flag = True

#       except:
#         pass

#     return Xb_sliced, Tb_dur, Tb_lat



def plot_learning_curve(d, spath=None):
  # axis
  x_tra = [d['bs_tra'] * d['log_interval'] * i for i in range(len(d['losses_dur_tra']))]
  y_dur_tra = d['losses_dur_tra']
  y_lat_tra = d['losses_lat_tra']
  y_isn_tra = d['losses_isn_tra']

  x_tes = np.arange(0, (1+d['max_epochs'])*d['n_tra'], d['n_tra'])[:-1]
  y_dur_tes = d['losses_dur_tes']
  y_lat_tes = d['losses_lat_tes']
  y_isn_tes = d['losses_isn_tes']

  fig, ax = plt.subplots(3, sharex='col')
  fig.suptitle('Learning Curves')
  # isnextRipple
  ax[0].plot(x_tra, y_isn_tra, color='blue')
  ax[0].scatter(x_tes, y_isn_tes, color='red')
  ax[0].legend(['Train Loss isnextRipple', 'Test Loss isnextRipple'], loc='upper right')
  # duration
  ax[1].plot(x_tra, y_dur_tra, color='blue')
  ax[1].scatter(x_tes, y_dur_tes, color='red')
  ax[1].legend(['Train Loss Duration', 'Test Loss Duration'], loc='upper right')
  # latency
  ax[2].plot(x_tra, y_lat_tra, color='blue')
  ax[2].scatter(x_tes, y_lat_tes, color='red')
  ax[2].legend(['Train Loss Latency', 'Test Loss Latency'], loc='upper right')
  ax[2].set_xlabel('Iteration (Total: {} epochs)'.format(d['epoch'][-1]))
  if spath:
    fig.savefig(spath)
    print('Saved to {}'.format(spath))
  fig.show()




## FIXED
def split_fpaths(fpaths, tes_keyword=None):
  idx_tra = [i for i, f in enumerate(fpaths) if f.find(tes_keyword) < 0]
  idx_tes = [i for i, f in enumerate(fpaths) if f.find(tes_keyword) > 0]
  fpaths_tra = [fpaths[i] for i in idx_tra]
  fpaths_tes = [fpaths[i] for i in idx_tes]
  return fpaths_tra, fpaths_tes

def load_an_lfp_and_a_rip(fpath_2kHz, samp_rate=1000): # input: npy, pkl (pands.DataFrame) -> output: torch.Tensor
  # Switch Load paths of lfp and rip by samp_rate
  if samp_rate == 2000:
    lpath_lfp = fpath_2kHz.replace('2kHz', '2kHz')
  if samp_rate == 1000:
    lpath_lfp = fpath_2kHz.replace('2kHz', '1kHz')
  if samp_rate == 500:
    lpath_lfp = fpath_2kHz.replace('2kHz', '500Hz')

  lpath_rip = fpath_2kHz.replace('2kHz', '1kHz').replace('.npy', '_rip_sec.pkl')
  # Load
  lfp = np.load(lpath_lfp).squeeze().astype(np.float32) # 2kHz -> int16, 1kHz, 500Hz -> float32
  print('Loaded :{}'.format(lpath_lfp))
  rip_sec = pkl_load(lpath_rip) # Pandas.DataFrame
  print('Loaded :{}'.format(lpath_rip))
  return lfp, rip_sec, samp_rate

def mk_dataset_from_an_lfp_and_a_rip(an_lfp, _a_rip_sec, samp_rate, max_seq_len, perturbation):
  a_rip_sec = _a_rip_sec.copy()
  assert perturbation <= max_seq_len
  last_rip_start = int(a_rip_sec.iloc[-1]['start_time'] * samp_rate)
  quotinent = int(last_rip_start / max_seq_len)
  reminder = last_rip_start % max_seq_len
  assert reminder + max_seq_len*quotinent == last_rip_start
  reference = reminder + max_seq_len
  start = reference - perturbation
  start_sec = start / samp_rate
  # align by start
  an_lfp = an_lfp[start:]
  a_rip_sec['start_time'] -= start_sec
  a_rip_sec['end_time'] -= start_sec
  a_rip_sec = a_rip_sec[0 < a_rip_sec['start_time']]

  Xb = []
  Tb_dur = []
  Tb_lat = []
  for i in range(quotinent-1):
    slice_start = i * max_seq_len
    slice_end = slice_start + max_seq_len
    slice_end_sec = slice_end / samp_rate
    lfp_sliced = an_lfp[slice_start:slice_end]
    higher_idx = bisect_left(a_rip_sec['start_time'].values, slice_end_sec)
    next_rip_dur = a_rip_sec['duration'].iloc[higher_idx]
    next_rip_lat = a_rip_sec['start_time'].iloc[higher_idx] - slice_end_sec
    Xb.append(lfp_sliced)
    Tb_dur.append(next_rip_dur)
    Tb_lat.append(next_rip_lat)
  Xb = np.vstack(Xb)
  Tb_dur = np.hstack(Tb_dur)
  Tb_lat = np.hstack(Tb_lat)
  return Xb, Tb_dur, Tb_lat


def test():
  lfp, rip_sec, samp_rate = load_an_lfp_and_a_rip(fpaths_tra[0], samp_rate=1000)
  Xb, Tb_dur, Tb_lat = mk_dataset_from_an_lfp_and_a_rip(lfp, rip_sec, 1000, 1000, True)


def wavelet(wave, samp_rate, f_min=100, f_max=None, plot=False):
  dt = 1. / samp_rate
  npts = len(wave)
  t = np.linspace(0, dt * npts, npts)
  if f_min == None:
      f_min = 0.1
  if f_max == None:
      f_max = int(samp_rate/2)
  scalogram = cwt(wave, dt, 8, f_min, f_max)

  if plot:
      fig = plt.figure()
      ax = fig.add_subplot(111)
      x, y = np.meshgrid(
          t,
          np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))

      ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
      ax.set_xlabel("Time [s]")
      ax.set_ylabel("Frequency [Hz]")
      ax.set_yscale('log')
      ax.set_ylim(f_min, f_max)
      plt.show()

  Hz = pd.Series(np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
  df = pd.DataFrame(np.abs(scalogram))
  df['Hz'] = Hz
  df.set_index('Hz', inplace=True)

  return df

# def wavelet_transform(wave, samp_rate, omega0=8): # Fixme
# from swan import pycwt # http://regeirk.github.io/pycwt/pycwt.html
#   freqs = np.arange((samp_rate/2)+1)
#   freqs[0] = 0.1
#   # Wavelet transformation
#   rr = np.abs(pycwt.cwt_f(wave, freqs, samp_rate, pycwt.Morlet(omega0)))
#   # rr.shape == (len(freqs), len(y))
#   return rr


def fft(x, f_s, normalization=False):
  global pd, fftpack
  if 'pd' not in locals():
    import pandas as pd
  if 'fftpack' not in locals():
    from scipy import fftpack

  X = fftpack.fft(x)
  freqs = fftpack.fftfreq(len(x)) * f_s
  # fig, ax = plt.subplots()
  # ax.stem(freqs, np.abs(X))
  # ax.set_xlabel('Frequency in Hertz [Hz]')
  # ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
  # ax.set_xlim(-f_s / 2, f_s / 2)
  index = [(0<=freqs)*(freqs<int(f_s/2))]
  powers = np.abs(X)[tuple(index)]
  if normalization:
    powers /= powers.sum() # normalize
  freqs = freqs[tuple(index)]
  out = pd.DataFrame({'powers':powers, 'Hz':freqs})
  out.set_index('Hz', inplace=True)
  # index = pd.Series({'freqs':freqs})
  # out = out.set_index(index)
  return out
  # ax.set_ylim(-5, 110)



def to_onehot(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def calc_weight(Tb, n_class):
  weights = torch.zeros_like(Tb).to(torch.float)
  recip_probs = torch.zeros_like(Tb).to(torch.float)
  counts = torch.zeros(n_class)

  for i in range(n_class):
    counts[i] += (Tb == i).sum()
  probs = counts / counts.sum()
  recip_probs = probs ** (-1)

  for i in range(n_class):
    mask = (Tb == i)
    weights[mask] = recip_probs[i]

  weights /= weights.mean() # normalize for stabilizing loss across batches
  return weights


def cm2df(cm, labels):
  df = pd.DataFrame(cm, \
                    index=['true:{}'.format(labels[0]), 'true:{}'.format(labels[1])], \
                    columns=['pred:{}'.format(labels[0]), 'pred:{}'.format(labels[1])])
  return df

def calc_confusion_matrix(y_true, y_pred, labels=['0','1'], normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass
    cm_df = cm2df(cm, labels)
    return cm_df

def plot_confusion_matrix(y_true, y_pred, classes=['0','1'],
                          normalize=False, title=None,
                          cmap=plt.cm.Blues, spath=None):

    from sklearn.metrics import confusion_matrix
    import pandas as pd

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix w/o norm.'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(title)
    else:
        print(title)
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if spath:
        dirname, fname, ext = split_fpath(spath)
        spath_root = dirname + fname

        spath_png = spath_root + '.png'
        fig.savefig(spath_png)
        print('Saved to {}'.format(spath_png))

        import pandas as pd
        spath_df = spath_root + '.csv'
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(spath_df)
        print('Saved to {}'.format(spath_df))
    return cm_df, ax

class class_recorder(): # for thesis
  def __init__(self):
    self.targets = []
    self.outputs = []

  def concat(self):
    self.targets = np.hstack(self.targets)
    self.outputs = np.hstack(self.outputs)

  def add_target(self, target):
    self.targets.append(target)

  def add_output(self, output):
    self.outputs.append(output)

  def conf_mat(self, labels=None):
    self.concat()
    return calc_confusion_matrix(self.targets, self.outputs, labels=labels)

  def cls_rep(self, labels=None):
    self.concat()
    return report_classification_results(self.targets, self.outputs, labels=labels)

  def report(self, labels_cm=None, labels_cr=None):
    self.concat()
    conf_mat = self.conf_mat(labels=labels_cm)
    cls_report = self.cls_rep(labels=labels_cr)
    self.__init__()
    return conf_mat, cls_report

# class class_recorder():
#   def __init__(self):
#     self.targets = []
#     self.outputs_cls = []
#     self.outputs_prob = []

#   def concat(self):
#     self.targets = np.hstack(self.targets)
#     self.outputs_cls = np.hstack(self.outputs_cls)
#     self.outputs_prob = np.hstack(self.outputs_prob)

#   def add_target(self, target):
#     self.targets.append(target)

#   def add_output_cls(self, output_cls):
#     self.outputs_cls.append(output_cls)

#   def add_output_prob(self, output_prob):
#     self.outputs_prob.append(output_prob)

#   def conf_mat(self, labels=None):
#     self.concat()
#     return calc_confusion_matrix(self.targets, self.outputs_cls, labels=labels)

#   def cls_rep(self, labels=None):
#     self.concat()
#     return report_classification_results(self.targets, self.outputs_cls, labels=labels)

#   def report(self, labels_cm=None, labels_cr=None):
#     self.concat()
#     conf_mat = self.conf_mat(labels=labels_cm)
#     cls_report = self.cls_rep(labels=labels_cr)
#     self.__init__()
#     return conf_mat, cls_report


def report_classification_results(y_true, y_pred, labels=[0,1], spath=None):
  d = classification_report(y_true, y_pred, labels=labels, output_dict=True)
  df = pd.DataFrame(d)
  # print(df)
  if spath:
    df.to_csv(spath)
    print('Saved to {}'.format(spath))
  return df

def calc_roc_curve(y_true, y_score, spath=None, plot=False):
  fpr, tpr, thresholds = roc_curve(y_true, y_score)
  roc_auc = auc(fpr, tpr)

  if plot:
      plt.figure()
      lw = 2
      plt.plot(fpr, tpr, lw=1, label='ROC curve (area = %0.2f)' % roc_auc, marker='o') # color='darkorange',
      plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver operating characteristic')
      plt.legend(loc="lower right")
      plt.show()
      if spath:
        plt.savefig(spath)
        print('Saved to {}'.format(spath))

  return fpr, tpr, thresholds, roc_auc

def calc_precision_recall_curve(y_true, probas_pred, pos_label=None, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true, probas_pred, pos_label=pos_label)
    thresholds = np.concatenate([np.array([0]), thresholds])
    pr_auc = auc(recall, precision)
    if plot:
        plt.plot(recall, precision, label='PRE-REC curve (area = %0.2f)' % pr_auc, marker='o') # color='darkorange',)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Thresholds {}'.format(thresholds))
        plt.legend(loc='lower left')
    return precision, recall, thresholds, pr_auc

def generate_random_str(n_str=8):
    import random
    import string
    # Generate a random string with n_str=8 characters.
    random_str = ''.join([random.choice(string.ascii_letters
               + string.digits) for n in range(n_str)])
    return random_str
