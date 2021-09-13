#!/usr/bin/env bash

ORIG_DIR=`pwd` #

cd ./data/CRCN_batch_download/download/

## Set your CRCN account
echo -e "\nPlease sign in CRCN.org (https://crcns.org/) and get your own account. Then, set the ID and PW in the ./data/batch_download/downaload/crcn-account.txt\n"
# cp ./data/batch_download/download/crcns-account.txt ./data/batch_download/download/crcns-account.txt_orig
$EDITOR ./crcns-account.txt


pwd
chmod u+x *.sh
./download.sh th-1

echo -e "\nPlease add + or # signs at the top of each line to determine which files to download.\n"
$EDITOR th-1/filelist.txt

## Downalods
./download.sh th-1 # y
./verify.sh th-1 # y

## Extracts tar.gz
cd th-1
cd data
tar xvf ./PreprocessedDatasets.tar.gz
cd dataset_2017_08_23
tar xvf ./2017-08-23_09-42-01.tar.gz
tar xvf ./2017-08-23_09-42-01_VT1_mpg.tar
tar xvf ./falcon_files.tar.gz


## mv th-1/* to ./data/ directory as ./data/th-1/*
cd $ORIG_DIR/data/CRCN_batch_download/download
mv th-1 $ORIG_DIR/data
cd $ORIG_DIR/data

## back to the original directory
cd $ORIG_DIR

## EOF
