#!/usr/bin/env bash

SAND_PATH=./singularity/sand_01
singularity build --sandbox --fakeroot $SAND_PATH $SAND_PATH.def


singularity shell -w $SAND_PATH
singularity shell --nv $SAND_PATH

## Locale; en_US.UTF-8
# https://www.tecmint.com/fix-failed-to-set-locale-defaulting-to-c-utf-8-in-centos/
yum -y install langpacks-en glibc-all-langpacks
export LC_ALL=en_US.UTF-8 # C # https://github.com/hpcng/singularity/issues/11
echo export LC_ALL=$LC_ALL >> $SINGULARITY_ENVIRONMENT

export LC_ALL=
