#!/bin/bash

#SBATCH --job-name=prepare-data
#SBATCH --output=prepare-data-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00 
#SBATCH --mem=64g

. ~/.bashrc
echo $TMPDIR

conda activate vs

cd ..

python src/prepare_data.py --base_path data

conda deactivate
cd script