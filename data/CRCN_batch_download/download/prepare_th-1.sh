#!/usr/bin/env bash

# ./data/CRCN_batch_download/download/prepare_th-1.sh

ORIG_DIR=`pwd`

cd ./data/CRCN_batch_download/download/

## Set your CRCN account
echo -e "\nPlease sign in CRCN.org (https://crcns.org/) and get your own account. Then, set the ID and PW in the ./data/batch_download/downaload/crcn-account.txt\n"

## Creates crcns-account.txt
if [ ! -f "crcns-account.txt" ]; then
    rm ./crcns-account.txt
fi

# echo 'file not exists'
# cp ./crcns-account.txt_orig ./crcns-account.txt
# $EDITOR ./crcns-account.txt

echo -e "----------------------------------------"
echo -e "\nCreating ./crcns-account.txt\n"    
echo -e "\ncrcns_username?: "
read CRCNS_USERNAME &&
echo crcns_username=\'${CRCNS_USERNAME}\' > ./crcns-account.txt

echo -e "\ncrcns_PASSWORD?: "
read CRCNS_PASSWORD &&
echo crcns_password=\'$CRCNS_PASSWORD\' >> ./crcns-account.txt
echo -e "----------------------------------------"

echo -e "$ cat ./crcns-account.txt"
cat ./crcns-account.txt
# fi

pwd
chmod u+x *.sh
./download.sh th-1

echo -e "\nPlease add + or # signs at the top of each line to determine which files to download. In the demo, \"data/Mouse12-120806.tar.gz\" was used.\n"
$EDITOR th-1/filelist.txt

## Downalods
./download.sh th-1 # y
./verify.sh th-1 # y



## Extracts tar.gz
cd th-1/data
tar -xvf Mouse12-120806.tar.gz
cd ../../../../th-1
ln -s ../CRCN_batch_download/download/th-1/data/

ls data/Mouse12-120806/Mouse12-120806.eeg
ls data/Mouse12-120806/Mouse12-120806.xml

cd $ORIG_DIR

# ## EOF
