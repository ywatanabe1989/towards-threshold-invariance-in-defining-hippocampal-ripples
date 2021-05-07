#!/usr/bin/env bash

rm -r $0.log

for nm in 01 02 03 04 05; do
    python3 ./ripples/define_ripples/using_CNN/checks_labels.py -nm $nm 2>&1 \
        | tee -a $0.log
done

# ./ripples/define_ripples/using_CNN/sh_scripts/checks_labels.sh

## EOF

