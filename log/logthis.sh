#!/usr/bin/env bash

logthis() {
    TIME=`date +"%Y-%m-%d-%H:%M"`
    COMMAND=`echo $@ | tr "/" "|"`
    LOG_FILE_NAME="${TIME}: $ $COMMAND.log"    
    echo $LOG_FILE_NAME
    $@ 2>&1 | tee -a log/"$LOG_FILE_NAME"
}

## EOF

