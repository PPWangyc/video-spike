#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="cebra"
#SBATCH --output="cebra.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 0-10:00
#SBATCH --export=ALL

. ~/.bashrc

eid=$1

cd ..
conda activate vs
python src/use_cebra.py --model_config config/model/linear_me.yaml \
                    --train_config config/train/rrr.yaml \
                    --eid $eid

cd script
conda deactivate
