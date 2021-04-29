#!/usr/bin/env python

import argparse

import sys; sys.path.append('.')
import utils.general as ug

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-p", "--fpath_pkl",
                default='./data/okada/01/day1/split/ripples_1kHz_csv/candi_orig/tt2-1_fp16.pkl', \
                help="The path of the input ripple file (.pkl)")
args = ap.parse_args()


fpath_pkl = args.fpath_pkl

rip_df = ug.load(fpath_pkl)

spath_csv = fpath_pkl.replace('.pkl', '.csv')

# rip_df.to_csv(spath_csv)
# print('Saved to: {}'.format(spath_csv))

ug.save(rip_df, spath_csv)

## EOF

# lpath_csv = spath_csv
# ug.load(lpath_csv).set_index('ripple_number')
