#!/usr/bin/env bash

## D0?+
# for nm in 01 02 03 04 05; do
for nm in 02 03 04; do    
    python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm -i 2>&1 \
        | tee -a $0_log/mouse_${nm}_i.log
done

## D0?-
# for nm in 01 02 03 04 05; do
for nm in 01 02 03 04; do    
    python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm 2>&1 \
        | tee -a $0_log/mouse_${nm}.log
done


# $ sc -S 'makes_labels1'
# $ sshell
# $ ./ripples/define_ripples/using_CNN/sh_scripts/makes_labels1.sh

## EOF

