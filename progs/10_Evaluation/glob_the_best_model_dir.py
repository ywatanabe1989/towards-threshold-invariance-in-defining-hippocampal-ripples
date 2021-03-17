import argparse
import myutils.myfunc as mf
import matplotlib.pyplot as plt
import numpy as np

def glob_the_last_model_dir(ts): # on ws
  # dirpath_root = '/mnt/nvme/Ripple_Detection/results/' + ts + '/' # fixme
  try:
      dirpath_root = '/mnt/md0/proj/results/' + ts + '/' # fixme
      dirpaths = mf.natsorted_glob(dirpath_root + 'epoch_*/batch_*/')
      dirpath_last = dirpaths[-1]
  except:
      dirpath_root = '../results/' + ts + '/' # fixme
      dirpaths = mf.natsorted_glob(dirpath_root + 'epoch_*/batch_*/')
      dirpath_last = dirpaths[-1]
  # dirpath_root = '../results/' + ts + '/' # fixme


  return dirpath_last, dirpaths

def glob_the_best_model_dir(ts, plot=False):

  d_path_last, dirpaths = glob_the_last_model_dir(ts)

  d = mf.load_pkl(d_path_last + 'data.pkl')
  # cr_reports = d['cls_isRipple_tra']
  cr_reports = d['cls_report_tra']

  f1_macros = []
  precisions = []
  for i in range(len(cr_reports)):
    precisions.append(cr_reports[i]['1']['precision'])
    f1_macros.append(cr_reports[i]['macro avg']['f1-score'])


  max_precision = max(precisions)
  max_precision_dir = dirpaths[np.argmax(precisions)]
  max_f1_macro_avg = max(f1_macros)
  max_f1_macro_avg_dir = dirpaths[np.argmax(f1_macros)]


  if plot:
    ## Precision
    fig, ax = plt.subplots()
    ax.plot(precisions)
    ax.set_ylabel('The Precision_Ripple of the Training Data')
    ax.set_xlabel('The Num. of the mini-Batch Iter.')
    txt = 'Max Precision of Training Data: {:.2f}, The Best Model: {}'.format(max_precision, max_precision_dir)
    ax.set_title(txt)


    ## F1-score
    fig, ax = plt.subplots()
    ax.plot(f1_macros)
    ax.set_ylabel('The Macro Avg. F1-score of the Training Data')
    ax.set_xlabel('The Num. of the mini-Batch Iter.')
    txt = 'Max macro avg. F1-score of Training Data: {:.2f}, The Best Model: {}'.format(max_f1_macro_avg, max_f1_macro_avg_dir)
    ax.set_title(txt)

    plt.pause(5)

  '''
  ## Confirm
  d_max_f1_macro_avg = mf.load_pkl(max_f1_macro_avg_dir + 'data.pkl')
  d_max_f1_macro_avg['cr_isRipple_tra'][-1]
  '''
  return max_f1_macro_avg_dir, dirpaths

def glob_the_best_model_dir(ts, plot=False):

  d_path_last, dirpaths = glob_the_last_model_dir(ts)

  d = mf.load_pkl(d_path_last + 'data.pkl')
  # cr_reports = d['cls_isRipple_tra']
  cr_reports = d['cls_report_tra']

  f1_macros = []
  precisions = []
  for i in range(len(cr_reports)):
    precisions.append(cr_reports[i]['1']['precision'])
    f1_macros.append(cr_reports[i]['macro avg']['f1-score'])


  max_precision = max(precisions)
  max_precision_dir = dirpaths[np.argmax(precisions)]
  max_f1_macro_avg = max(f1_macros)
  max_f1_macro_avg_dir = dirpaths[np.argmax(f1_macros)]


  if plot:
    ## Precision
    fig, ax = plt.subplots()
    ax.plot(precisions)
    ax.set_ylabel('The Precision_Ripple of the Training Data')
    ax.set_xlabel('The Num. of the mini-Batch Iter.')
    txt = 'Max Precision of Training Data: {:.2f}, The Best Model: {}'.format(max_precision, max_precision_dir)
    ax.set_title(txt)


    ## F1-score
    fig, ax = plt.subplots()
    ax.plot(f1_macros)
    ax.set_ylabel('The Macro Avg. F1-score of the Training Data')
    ax.set_xlabel('The Num. of the mini-Batch Iter.')
    txt = 'Max macro avg. F1-score of Training Data: {:.2f}, The Best Model: {}'.format(max_f1_macro_avg, max_f1_macro_avg_dir)
    ax.set_title(txt)

    plt.pause(5)

  '''
  ## Confirm
  d_max_f1_macro_avg = mf.load_pkl(max_f1_macro_avg_dir + 'data.pkl')
  d_max_f1_macro_avg['cr_isRipple_tra'][-1]
  '''
  return max_f1_macro_avg_dir, dirpaths


if __name__ == '__main__':
  ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  ap.add_argument("-ts", "--timestamp", default='191022/230633', help = " ")
  args = ap.parse_args()

  ts = args.timestamp
  max_f1_macro_avg_dir, max_precision_dir, dirpaths = glob_the_best_model_dir(ts, plot=True)
