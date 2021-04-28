#!/usr/bin/env bash


cat data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt | \
xargs -P 20 -n 1 python3 ripples/EDA/candidates/rename_from_nas1.py -n

## EOF
# ./ripples/EDA/candidates/sh_scripts/rename_from_nas1.sh 
