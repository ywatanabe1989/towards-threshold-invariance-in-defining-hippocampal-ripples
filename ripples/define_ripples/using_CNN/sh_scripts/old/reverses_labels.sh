#!/usr/bin/env bash

for nm in 01 02 03 04 05; do
    for i in -i ''; do
        python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i
    done
done


# logthis sh ripples/define_ripples/using_CNN/sh_scripts/reverses_labels.sh



ls ./data/okada/0?/day?/split/ripples_1kHz_pkl/CNN_labeled/D0??/tt* | wc -l

for f in ./data/okada/0?/day?/split/ripples_1kHz_pkl/CNN_labeled/D0??/tt*; do
    rename tt _wrong_tt $f
done



ls ./data/okada/0?/day?/split/ripples_1kHz_pkl/CNN_labeled/D0??/reversed_tt* | wc -l

for f in ./data/okada/0?/day?/split/ripples_1kHz_pkl/CNN_labeled/D0??/reversed_tt*; do
    rename reversed_tt tt $f
done

ls ./data/okada/0?/day?/split/ripples_1kHz_pkl/CNN_labeled/D0??/tt* | wc -l

# ################################################################################
# ## D0?+
# ################################################################################
# nm=01; i=-i
# python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i

# nm=02; i=-i
# python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i

# nm=03; i=-i
# python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i

# nm=04; i=-i
# python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i

# nm=05; i=-i
# python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i


# ################################################################################
# ## D0?-
# ################################################################################
# nm=01; i=''
# python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i

# nm=02; i=''
# python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i

# nm=03; i=''
# python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i

# nm=04; i=''
# python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i

# nm=05; i=''
# python3 ./ripples/define_ripples/using_CNN/reverses_labels.py -nm $nm $i



## EOF

