#!/usr/bin/env bash

# Glob "./data/05/day4/split/1kHz_npy/1_tt?-?_fp16.npy" file paths and run mouse05_48h_to_2days.py on each of them.

N_CPUS=20

echo ./data/05/day4/split/1kHz_npy/1_tt?-?_fp16.npy | xargs \
-P $N_CPUS -n 1 ./progs/00_NSx2npy/mouse05_48h_to_2days.py -n


## EOF
