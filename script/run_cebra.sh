#!/bin/bash

mode=$1
while IFS= read -r line
do
    echo "Processing $line"
    sbatch use_cebra.sh $line
done < ../data/eid.txt