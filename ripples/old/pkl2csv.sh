#!/usr/bin/env bash

rm $0.log

# echo ./data/okada/0?/day?/split/ripples_1kHz_csv/*/*.pkl | xargs -P 20 -n 1 python3 ./ripples/pkl2csv.py -p


for f in ./data/okada/0?/day?/split/ripples_1kHz_csv/*/*.pkl; do
    sleep 1
    echo $f
    python3 ./ripples/pkl2csv.py -p $f
done


