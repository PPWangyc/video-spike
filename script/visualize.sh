#!/bin/bash

. ~/.bashrc
echo $TMPDIR


conda activate vs

cd ..

python src/visualize_result.py --log_dir /expanse/lustre/scratch/ywang74/temp_project/video-spike/results

conda deactivate
cd script