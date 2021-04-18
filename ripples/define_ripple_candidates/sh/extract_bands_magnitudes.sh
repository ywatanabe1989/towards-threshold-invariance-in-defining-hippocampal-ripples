#!/usr/bin/env bash


echo -e '\nRun ./extract_bands_magnitudes.py on each file written in $SEMI_RIPPLE_HOME/data/HIPPO_LFP_TT_IDs.txt.\n'

N_CPUS=18
FILE_NAME=./progs/Fig_01_02_Ripple_candidates/extract_bands_magnitudes

cat $SEMI_RIPPLE_HOME/data/HIPPO_LFP_TT_IDs.txt |
    xargs -P $N_CPUS -n 1 ${FILE_NAME}.py -l 2>&1 | \
    tee ${FILE_NAME}_sh.log
    

## EOF
