#!/usr/bin/env bash

# rm -r $0.log $0_log/
# mkdir $0_log

# for nm in 01 02 03 04 05; do
#     for i in -i ''; do
#     python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
#         | tee -a $0_log/mouse_${nm}_${i}.log
#     done
# done

for nm in 01; do
    for i in -i; do
    python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
        | tee -a $0_log/mouse_${nm}_${i}.log
    done
done

# ./ripples/define_ripples/using_CNN/sh_scripts/makes_labels.sh

## EOF

