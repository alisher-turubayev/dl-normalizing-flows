#!/bin/bash
DATASET_PATH="$1"

for FILE in "${DATASET_PATH}"*.jpg
do
    #FILENAME=$(echo $FILE | tr " " "\n");
    SIZE=$(identify -format '%w %h' $FILE | tr " " "\n");
    CURRENT_FILE_PASS=true
    for SIDE in $SIZE
    do 
        if [ "$SIDE" -lt "64" ] 
            then
            CURRENT_FILE_PASS=false
        fi
    done
    
    if [ $CURRENT_FILE_PASS == false ]
    then
        echo 'File deleted.'
        $(rm $FILE)
    fi
done
