#!/usr/bin/env bash

# Glob "./data/05/day4/split/1kHz_npy/1_tt?-?_fp16.npy" file paths and run mouse05_48h_to_2days.py on each of them.

N_CPUS=10

# echo ./data/0?/day?/split/1kHz_npy/*tt?-?_fp16.npy | \
# xargs -P $N_CPUS -n 1 ./progs/Fig_01_02_Ripple_candidates/detect_ripple_candidates.py -n



# echo ./data/0?/day?/split/1kHz_npy/*tt?-?_fp16.npy | \
# xargs -P $N_CPUS -n 1 ./progs/Fig_01_02_Ripple_candidates/detect_ripple_candidates.py -n | xargs tee ./progs/Fig_01_02_Ripple_candidates/detect_ripple_candidates_sh.log


echo ./data/0?/day?/split/1kHz_npy/*tt?-?_fp16.npy | \
    xargs -P $N_CPUS -n 1 ./progs/Fig_01_02_Ripple_candidates/detect_ripple_candidates.py -n 2>&1 | \
    tee ./progs/Fig_01_02_Ripple_candidates/detect_ripple_candidates_sh.log



# echo ./data/0?/day?/split/1kHz_npy/*tt?-?_fp16.npy | \
# xargs -L 1 -P $N_CPUS -n 1 sh -c './progs/Fig_01_02_Ripple_candidates/detect_ripple_candidates.py -n "$1"'

# echo ./data/0?/day?/split/1kHz_npy/*tt?-?_fp16.npy | \
#     parallel -k ./progs/Fig_01_02_Ripple_candidates/detect_ripple_candidates.py -n \
#              > ./progs/Fig_01_02_Ripple_candidates/detect_ripple_candidates_sh.log

## EOF
