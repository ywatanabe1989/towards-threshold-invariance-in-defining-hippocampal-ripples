#!/usr/bin/env bash

cat data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt | \
xargs -P 20 -n 1 python3 ripples/define_ripples/conventional/calc_props.py -n 2>&1 | tee $0.log

# ./ripples/define_ripples/conventional/sh_scripts/calc_props.sh

## EOF
