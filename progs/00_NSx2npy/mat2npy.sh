#!/usr/bin/env bash


# Glob "./data/0?/day?/split/2kHz_mat/tt?-?.mat" file paths and run mat2npy.py on each of them.


N_CPUS=20


echo ./data/0?/day?/split/2kHz_mat/*tt?-?.mat | xargs \
-P $N_CPUS -n 1 ./progs/00_NSx2npy/mat2npy.py --dtype np.float16 --tgt_samp_rate 1000 -l


## EOF
