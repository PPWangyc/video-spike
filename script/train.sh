#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="train"
#SBATCH --output="train.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 1-00
#SBATCH --export=ALL

. ~/.bashrc

cd ..

conda activate vs
input_mod=$1
python src/train.py --model_config config/model/linear_$input_mod.yaml \
                    --train_config config/train/linear_$input_mod.yaml 

cd script
conda deactivate
