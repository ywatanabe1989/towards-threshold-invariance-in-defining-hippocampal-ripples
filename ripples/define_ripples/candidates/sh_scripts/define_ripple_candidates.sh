#!/usr/bin/env bash


echo -e '\nRun ./mouse05_48h_to_2days.py on each file written in $SEMI_RIPPLE_HOME/data/HIPPO_LFP_TT_IDs.txt. There are 184 numpy files in total. It takes about 7 hours for our machine to perform the script on a numpy file. Thus, roughly (184 * 7 / N_CPUS) hours are required to pick all ripple candidates from our dataset. In default, N_CPUS are set as 18. Logs are written in ./progs/Fig_01_02_Ripple_candidates/detect_ripple_candidates_sh.log.\n'


N_CPUS=18

cat $SEMI_RIPPLE_HOME/data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt |
xargs -P $N_CPUS -n 1 python3 ./ripples/define_ripples/candidates/detects_ripple_candidates.py -n 2>&1 | tee $0.log

## EOF
