import numpy as np
import os

data_dir = 'data'
# get all files that start with data_rrr_cebra
files = [f for f in os.listdir(data_dir) if f.startswith('data_rrr_pca')]
print(files)
train_data = {}
for f in files:
    data = np.load(os.path.join(data_dir, f), allow_pickle=True).item()
    train_data.update(data)

# save the data
np.save(os.path.join(data_dir, 'data_rrr_pca.npy'), train_data)
