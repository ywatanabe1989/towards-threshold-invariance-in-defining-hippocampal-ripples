#!/usr/bin/env bash

for imt in 01 02 03 04 05; do
    echo python3 ./ripples/detect_ripples/CNN/predict_n_s_r.py -im $imt
    python3 ./ripples/detect_ripples/CNN/predict_n_s_r.py -im $imt
done

# ./ripples/detect_ripples/CNN/sh_scripts/predict_n_s_r.sh
## EOF
