#!/usr/bin/env bash


cat data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt | \
xargs -P 20 -n 1 python3 ripples/EDA/candidates/calc_props.py -n 2>&1 | tee $0.log

# ./ripples/EDA/candidates/sh_scripts/calc_props.sh

## EOF
