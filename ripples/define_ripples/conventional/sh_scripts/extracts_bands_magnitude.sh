#!/usr/bin/env bash

cat ./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt |
    xargs -P $N_CPUS -n 1 \
    python3 ripples/define_ripples/conventional/extracts_bands_magnitude.py -l

# logthis ./ripples/define_ripples/conventional/sh_scripts/extracts_bands_magnitude.sh

## EOF








# echo -e '\nRun ./extract_bands_magnitudes.py on each file written in $SEMI_RIPPLE_HOME/data/HIPPO_LFP_TT_IDs.txt.\n'

# N_CPUS=18
# FILE_NAME=./progs/Fig_01_02_Ripple_candidates/extracts_bands_magnitude

# cat $SEMI_RIPPLE_HOME/data/HIPPO_LFP_TT_IDs.txt |
#     xargs -P $N_CPUS -n 1 ${FILE_NAME}.py -l 2>&1 | \
#     tee ${FILE_NAME}_sh.log



# for f in `cat ./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt`; do


# cat $SEMI_RIPPLE_HOME/data/HIPPO_LFP_TT_IDs.txt |
#     xargs -P $N_CPUS -n 1 ${FILE_NAME}.py -l 2>&1 | \
#     tee ${FILE_NAME}_sh.log
