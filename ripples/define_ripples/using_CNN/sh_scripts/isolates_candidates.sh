#!/usr/bin/env bash

rm $0.log

cat ./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt \
    | xargs -P 20 -n 1 python3 ./ripples/define_ripples/using_CNN/isolates_candidates.py -l \
    | tee -a $0.log

# ./ripples/define_ripples/using_CNN/sh_scripts/isolates_candidates.sh


## EOF
