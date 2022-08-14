#!/bin/bash
DATASET_PATH="$1"

for FILE in $DATASET_PATH
do
    if [[ $FILE==*.jpg ]]
    then
        FILENAME=$(echo $FILE | tr " " "\n")

        SIZE=$(identify -format '%w %h' $FILENAME);
        SIZE_SPLIT=$(echo $SIZE | tr " " "\n");
        
        CURRENT_FILE_PASS=true
        for SIDE in $SIZE_SPLIT
        do 
            if [ "$SIDE" -lt "48" ] 
                then
                CURRENT_FILE_PASS=false
            fi
        done
        
        if [ $CURRENT_FILE_PASS == false ]
        then
            echo 'File deleted.'
            $(rm $FILE)
        fi
    fi
done
