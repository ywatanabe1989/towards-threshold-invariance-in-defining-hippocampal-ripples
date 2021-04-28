#!/usr/bin/env bash


for nm in 01 02 03 04 05; do
    for s in png mp4; do
        python3 ./ripples/define_ripples/conventional/plots_3d_scatter.py -nm $nm -s $s 2>&1 | tee $0.log
    done
done

# ./ripples/define_ripples/conventional/sh_scripts/plots_3d_scatter.sh

## EOF
