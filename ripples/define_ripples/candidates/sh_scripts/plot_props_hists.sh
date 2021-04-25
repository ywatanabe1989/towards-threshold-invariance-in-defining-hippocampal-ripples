#!/usr/bin/env bash


echo 01 02 03 04 05 | \
xargs -P 20 -n 1 python3 ripples/EDA/candidates/plot_props_hists.py -n 2>&1 | tee $0.log

# ./ripples/EDA/candidates/sh_scripts/plot_props_hists.sh

## EOF
