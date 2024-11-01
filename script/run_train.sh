#!/bin/bash

mode=$1
while IFS= read -r line
do
    echo "Processing $line"
    sbatch train,sh $mode $line
done < ../data/eid.txt