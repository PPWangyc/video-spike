#!/bin/bash

model=$1
while IFS= read -r line
do
    echo "Processing $line $model"
    sbatch pretrain.sh $line $model
done < ../data/eid.txt