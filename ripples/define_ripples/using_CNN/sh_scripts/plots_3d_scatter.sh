#!/usr/bin/env bash

rm $0.log

for nm in 01 02 03 04 05; do
    # for i in '' -i; do
    for i in -i; do # fixme
        for s in png mp4; do
            python3 ./ripples/define_ripples/using_CNN/plots_3d_scatter.py -nm $nm -s $s $i 2>&1 \
                | tee -a $0.log
        done
    done
done


# ./ripples/define_ripples/using_CNN/sh_scripts/plots_3d_scatter.sh

## EOF
