#!/usr/bin/env bash

rm $0.log

for i in -i ''; do
    for nm in 01 02 03 04 05; do
        python3 ./ripples/define_ripples/using_GMM/makes_labels.py -nm $nm $i 2>&1 | tee -a $0.log
    done
done

# ./ripples/define_ripples/using_GMM/sh_scripts/makes_labels.sh

## EOF

