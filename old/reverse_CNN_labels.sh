#!/usr/bin/env bash


ls ./data/okada/0?/day?/split/ripples_1kHz_pkl/CNN_labeled/D0??/tt* | wc -l

ls ./data/okada/0?/day?/split/ripples_1kHz_pkl/CNN_labeled/D0??/reversed_tt* | wc -l



for f in ./data/okada/0?/day?/split/ripples_1kHz_pkl/CNN_labeled/D0??/tt*; do
    rename tt _wrong_tt $f
done


for f in ./data/okada/0?/day?/split/ripples_1kHz_pkl/CNN_labeled/D0??/reversed_tt*; do
    rename reversed_tt tt $f
done
