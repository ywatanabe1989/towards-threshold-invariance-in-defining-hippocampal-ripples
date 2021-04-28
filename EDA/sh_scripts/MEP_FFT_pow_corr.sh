#!/usr/bin/env bash

rm $0.log

for nm in 01 02 03 04 05; do
    python3 ./EDA/MEP_FFT_pow_corr.py -nm $nm | tee -a $0.log
done

# ./EDA/sh_scripts/MEP_FFT_pow_corr.sh

## EOF
