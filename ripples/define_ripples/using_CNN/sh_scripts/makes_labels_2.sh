#!/usr/bin/env bash

rm -r $0.log $0_log
mkdir $0_log

################################################################################
## D0?+
################################################################################
nm=01; i=-i
python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
    | tee -a $0_log/mouse_${nm}_${i}.log

nm=02; i=-i
python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
    | tee -a $0_log/mouse_${nm}_${i}.log

nm=03; i=-i
python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
    | tee -a $0_log/mouse_${nm}_${i}.log

nm=04; i=-i
python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
    | tee -a $0_log/mouse_${nm}_${i}.log

nm=05; i=-i
python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
    | tee -a $0_log/mouse_${nm}_${i}.log


sleep 600

################################################################################
## D0?-
################################################################################
nm=01; i=''
python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
    | tee -a $0_log/mouse_${nm}_${i}.log

nm=02; i=''
python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
    | tee -a $0_log/mouse_${nm}_${i}.log

nm=03; i=''
python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
    | tee -a $0_log/mouse_${nm}_${i}.log

nm=04; i=''
python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
    | tee -a $0_log/mouse_${nm}_${i}.log

nm=05; i=''
python3 ./ripples/define_ripples/using_CNN/makes_labels.py -nm $nm $i 2>&1 \
    | tee -a $0_log/mouse_${nm}_${i}.log


# ./ripples/define_ripples/using_CNN/sh_scripts/makes_labels_2.sh

## EOF

