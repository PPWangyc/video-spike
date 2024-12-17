#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="create_eid_data"
#SBATCH --output="create_eid_data.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 0-2:00
#SBATCH --export=ALL

. ~/.bashrc

cd ..
input_mod=$1
conda activate vs
python src/create_eid_data.py --model_config config/model/linear_me.yaml \
                    --train_config config/train/rrr.yaml \
                    --input_mod $input_mod

cd script
conda deactivate
