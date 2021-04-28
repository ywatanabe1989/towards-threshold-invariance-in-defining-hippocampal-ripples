#!/usr/bin/env bash


for nm in 01 02 03 04 05; do
    python3 ./ripples/define_ripples/using_GMM/make_labels.py -nm $nm 2>&1 | tee $0.log
done


# ./ripples/define_ripples/using_GMM/sh_scripts/make_labels.sh
## EOF

