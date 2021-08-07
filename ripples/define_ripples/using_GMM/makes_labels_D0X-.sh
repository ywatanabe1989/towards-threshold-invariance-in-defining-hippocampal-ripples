#!/usr/bin/env bash

'''
Since GMM clustering should be run on each mouse, D0X- is not defined for GMM_labeled.
However, for consistency, this code snipet softlinks from D0X- to D0X+.
'''
RIPPLE_PROJ_HOME=/mnt/nvme/proj/Semisupervised_Ripples

################################################################################
################################################################################
n_mouse=01
for n_mouse_src in 02 03 04 05; do
    for f in $RIPPLE_PROJ_HOME/data/okada/$n_mouse/day?/split/ripples_1kHz_pkl/GMM_labeled/D${n_mouse}+; do
        cd $f
        cd ..
        ln -s D${n_mouse}+ D${n_mouse_src}-
    done
done
################################################################################
################################################################################
n_mouse=02
for n_mouse_src in 01 03 04 05; do
    for f in $RIPPLE_PROJ_HOME/data/okada/$n_mouse/day?/split/ripples_1kHz_pkl/GMM_labeled/D${n_mouse}+; do
        cd $f
        cd ..
        ln -s D${n_mouse}+ D${n_mouse_src}-
    done
done
################################################################################
################################################################################
n_mouse=03
for n_mouse_src in 01 02 04 05; do
    for f in $RIPPLE_PROJ_HOME/data/okada/$n_mouse/day?/split/ripples_1kHz_pkl/GMM_labeled/D${n_mouse}+; do
        cd $f
        cd ..
        ln -s D${n_mouse}+ D${n_mouse_src}-
    done
done
################################################################################
################################################################################
n_mouse=04
for n_mouse_src in 01 02 03 05; do
    for f in $RIPPLE_PROJ_HOME/data/okada/$n_mouse/day?/split/ripples_1kHz_pkl/GMM_labeled/D${n_mouse}+; do
        cd $f
        cd ..
        ln -s D${n_mouse}+ D${n_mouse_src}-
    done
done
################################################################################
################################################################################
n_mouse=05
for n_mouse_src in 01 02 03 04; do
    for f in $RIPPLE_PROJ_HOME/data/okada/$n_mouse/day?/split/ripples_1kHz_pkl/GMM_labeled/D${n_mouse}+; do
        cd $f
        cd ..
        ln -s D${n_mouse}+ D${n_mouse_src}-
    done
done
################################################################################
################################################################################
