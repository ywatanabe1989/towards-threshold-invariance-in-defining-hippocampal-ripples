#!/usr/bin/env bash

for nm in 01 02 03 04 05; do
    for dur in 3 5 10; do
        python3 ./ripples/define_ripples/using_CNN/checks_traces.py -nm $nm -dur $dur 2>&1 \
            | tee -a $0.log
    done
done

# ./ripples/define_ripples/using_CNN/sh_scripts/checks_traces.sh

## EOF

