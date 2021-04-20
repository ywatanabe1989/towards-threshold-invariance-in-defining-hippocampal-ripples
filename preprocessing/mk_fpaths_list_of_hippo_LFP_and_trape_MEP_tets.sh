#!/bin/bash

SAMP_RATE_STR=1kHz

## Hippocampal LFP list
OUT_FILE=$SEMI_RIPPLE_HOME/data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt
rm $OUT_FILE
for i_mouse in `seq 5`; do
    HIPPO_LFP_TTs=`cat ./data/okada/0$i_mouse/MouseInfo_en.md | tail -n 2 | head -n 1 | \
                   cut -d ":" -f 1 | cut -d "t" -f 3 | tr -d ,`

    for i_tt in $HIPPO_LFP_TTs; do
        ls ./data/okada/0$i_mouse/day?/split/LFP_MEP_${SAMP_RATE_STR}_npy/orig/tt${i_tt}-?_fp16.npy \
             | xargs -n 1 echo >> $OUT_FILE
    done
    
done
echo Created: $OUT_FILE


## Trapezius MEP list
OUT_FILE=$SEMI_RIPPLE_HOME/data/okada/FPATH_LISTS/TRAPE_MEP_TT_NPYs.txt
rm $OUT_FILE
for i_mouse in `seq 5`; do
    TRAPE_MEP_TTs=`cat ./data/okada/0$i_mouse/MouseInfo_en.md | tail -n 1 | \
                   cut -d ":" -f 1 | cut -d "t" -f 3 | tr -d ,`

    for i_tt in $TRAPE_MEP_TTs; do
        echo ./data/okada/0$i_mouse/day?/split/LFP_MEP_${SAMP_RATE_STR}_npy/orig/tt${i_tt}-?_fp16.npy \
             | xargs -n 1 echo >> $OUT_FILE
    done
    
done
echo Created: $OUT_FILE

## EOF
