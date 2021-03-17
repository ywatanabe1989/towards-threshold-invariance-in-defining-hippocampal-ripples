import sys
sys.path.append('./')
sys.path.append('./06_File_IO')
sys.path.append('./07_Learning/')
sys.path.append('./11_Models/')
import utils.myfunc as mf
import argparse
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-ts", "--timestamp", help="Time stamp to resume (ex. 190830/114015)", type=str)
ap.add_argument("-e", "--epoch", help="Epoch to resume", type=int)
args = ap.parse_args()


p = mf.pkl_load('../results/{}/epoch_{}/params.pkl'.format(args.timestamp, args.epoch))
print(p['n_load'])
d = mf.pkl_load('../results/{}/epoch_{}/data.pkl'.format(args.timestamp, args.epoch))
try:
  print(d['class_report_tes'])
except:
  pass

import matplotlib.pyplot as plt
plt.plot(d['losses_acc_tes'])
