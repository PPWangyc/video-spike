#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="pretrain"
#SBATCH --output="pretrain.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 0-20:00
#SBATCH --export=ALL

. ~/.bashrc

cd ..

conda activate vs
eid=$1
model=$2 # c, m, cm; contrast, mask, contrast_mask
accelerate launch --config_file config/accelerate/default.yaml src/pretrain.py --model_config config/model/vit_mae/vit_mae.yaml \
                    --train_config config/train/vmae_video.yaml \
                    --model $model \
                    --eid $eid
# python src/pretrain.py --model_config config/model/vit_mae/vit_mae.yaml \
#                     --train_config config/train/vmae_video.yaml \
#                     --eid $eid \
#                     --model $model
cd script
conda deactivate
