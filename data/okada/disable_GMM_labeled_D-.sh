#!/usr/bin/env bash

for f in ./data/okada/0?/day?/split/ripples_1kHz_pkl/GMM_labeled/D0?-; do
    rename D0 _D0 $f --verbose
done

## EOF
    
