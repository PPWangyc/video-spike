import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--label', type=str, default='cebra', help='label for the data')

args = parser.parse_args()

data_dir = 'data'
label = args.label
# get all files that start with data_rrr_cebra
files = [f for f in os.listdir(data_dir) if f.startswith(f'data_rrr_{label}_')]
print(files)
train_data = {}
for f in files:
    data = np.load(os.path.join(data_dir, f), allow_pickle=True).item()
    train_data.update(data)

# save the data
np.save(os.path.join(data_dir, f'data_rrr_{label}.npy'), train_data)
