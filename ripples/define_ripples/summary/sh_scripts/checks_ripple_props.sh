#!/usr/bin/env bash

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

for nm in 01 02 03 04 05; do
    for ftr in "duration" "mep" 'ripple peak magnitude'; do
        echo 'python3 ./ripples/define_ripples/summary/checks_ripple_props.py -nm $nm -ftr $ftr 2>&1 | tee -a $0.log'
        python3 ./ripples/define_ripples/summary/checks_ripple_props.py -nm $nm -ftr $ftr 2>&1 | tee -a $0.log
    done
done

# ./ripples/define_ripples/summary/sh_scripts/checks_ripple_props.sh

## EOF
