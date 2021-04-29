#!/usr/bin/env bash

rm $0.log

for nm in 01 02 03 04 05; do
    for s in png mp4; do
        python3 ./ripples/define_ripples/using_GMM/plots_3d_scatter.py -nm $nm -s $s 2>&1 | tee -a $0.log
    done
done


# ./ripples/define_ripples/using_GMM/sh_scripts/plots_3d_scatter.sh
## EOF

