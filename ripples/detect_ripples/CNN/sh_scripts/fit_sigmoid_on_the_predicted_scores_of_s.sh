#!/usr/bin/env bash

for imt in 01 02 03 04 05; do
    echo python3 ./ripples/detect_ripples/CNN/fit_sigmoid_on_the_predicted_scores_of_s.py -im $imt
    python3 ./ripples/detect_ripples/CNN/fit_sigmoid_on_the_predicted_scores_of_s.py -im $imt
done

# ./ripples/detect_ripples/CNN/sh_scripts/fit_sigmoid_on_the_predicted_scores_of_s.sh
## EOF
