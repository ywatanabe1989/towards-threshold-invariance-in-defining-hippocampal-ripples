#!/usr/bin/env bash


echo 01 02 03 04 05 | \
xargs -P 20 -n 1 python3 ripples/define_ripples/conventional/plots_prop_hists.py -n 2>&1 | tee $0.log

# ./ripples/define_ripples/conventional/sh_scripts/plots_prop_hists.sh

## EOF
