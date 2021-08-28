#!/usr/bin/env bash

for nm in 01 02 03 04 05; do
    python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm -i 2>&1 \
        | tee -a $0_log/mouse_-i.log
done

# $ sc -S 'makes_labels2'
# $ sshell
# $ ./ripples/define_ripples/using_CNN/sh_scripts/makes_labels2.sh

## EOF
