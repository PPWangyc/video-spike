from utils.utils import (
    get_args,
    set_seed,
    get_rrr_data,
    get_cebra_embedding,
    get_pca_embedding,
    NAME2MODEL
)
from utils.dataset_utils import (
    split_dataset,
    get_metadata_from_loader
)
from utils.utils import (
    _std
)
from utils.metric_utils import (
    bits_per_spike,
)
from utils.config_utils import (
    config_from_kwargs,
    update_config
)
from loader.make import (
    make_loader
)
import cebra
import numpy as np
from tqdm import tqdm

# set config
args = get_args()
kwargs = {"model": "include:{}".format(args.model_config)}
config = config_from_kwargs(kwargs)
config = update_config(args.train_config, config)
config = update_config(args, config)
# set seed
set_seed(config.seed)
use_pca = False
label = 'cebra' if not use_pca else 'pca'
eid = args.eid
dataset_split_dict = split_dataset(config.dirs.data_dir,eid=eid)
train_dataloader, val_dataloader, test_dataloader = make_loader(config, dataset_split_dict)
meta_data = get_metadata_from_loader(test_dataloader, config)
print(f"meta_data: {meta_data}")

train_data = {
        eid:
        {
            "X": [], 
            "y": [], 
            "setup": {}
        } 
        for eid in [eid]
    }
out_dim = 5

# Cebra embeddings
# get train whisker-video
train_X , train_y = get_rrr_data(train_dataloader, 'whisker-video')
# append neural activity to train data as y
train_data[eid]["y"].append(train_y)

# Test data
test_X, test_y = get_rrr_data(test_dataloader, 'whisker-video')
train_data[eid]["y"].append(test_y)

all_X = np.concatenate([train_X, test_X], axis=0)
train_idx, test_idx = np.arange(train_X.shape[0]), np.arange(train_X.shape[0], all_X.shape[0])
print(all_X.shape)
# get cebra embeddings
save_path = f"{label}_{eid[:5]}"
all_X = get_pca_embedding(all_X, out_dim=out_dim) if use_pca else get_cebra_embedding(all_X, out_dim=out_dim,save_path=save_path)
train_X = all_X[train_idx]
test_X = all_X[test_idx]
train_data[eid]["X"].append(train_X)
train_data[eid]["X"].append(test_X)
# cebra_train = get_cebra_embedding(train_X, out_dim=out_dim)
# # append cebra embeddings to train data as X
# train_data[eid]["X"].append(cebra_train)

# Test data
# test_X, test_y = get_rrr_data(test_dataloader, 'whisker-video')
# train_data[eid]["y"].append(test_y)
# pca_test = get_pca_embedding(test_X, out_dim=out_dim)
# train_data[eid]["X"].append(pca_test)
# cebra_test = get_cebra_embedding(test_X, out_dim=out_dim)
# train_data[eid]["X"].append(cebra_test)

print(train_data[eid]["X"][0].shape)
print(train_data[eid]["y"][0].shape)
print(train_data[eid]["X"][1].shape)
print(train_data[eid]["y"][1].shape)


# save data
# np.save(f"data/data_rrr_cebra_{eid[:5]}.npy", train_data)
np.save(f"data/data_rrr_{label}_{eid[:5]}.npy", train_data)