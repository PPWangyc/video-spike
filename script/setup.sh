#!/bin/bash

. ~/.bashrc
echo $TMPDIR

# module load
ml git-lfs/2.11.0
git lfs install

# clone dataset
cd ../data
git clone git@hf.co:datasets/PPWangyc/ibl-video

cd ..

# create conda environment
conda env create -f env.yaml

cd script
