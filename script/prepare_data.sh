#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="prepare_data"
#SBATCH --output="prepare_data.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 5-00
#SBATCH --export=ALL

. ~/.bashrc
echo $TMPDIR


conda activate vs

cd ..

python src/prepare_data.py --base_path data

# # module load
# ml git-lfs/2.11.0
# cd data/ibl-video
# git lfs install
# # track *tar files
# git lfs track "*.tar"
# git add .
# git commit -m "track *tar files"
# git push
# cd ../..
conda deactivate
cd script