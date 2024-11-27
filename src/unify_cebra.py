import numpy as np
import os

data_dir = 'data'
use_pca = False
label = 'cebra' if not use_pca else 'pca'
# get all files that start with data_rrr_cebra
files = [f for f in os.listdir(data_dir) if f.startswith(f'data_rrr_{label}')]
print(files)
train_data = {}
for f in files:
    data = np.load(os.path.join(data_dir, f), allow_pickle=True).item()
    train_data.update(data)

# save the data
np.save(os.path.join(data_dir, f'data_rrr_{label}.npy'), train_data)
