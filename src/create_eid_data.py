from utils.utils import (
    get_args,
    set_seed,
    get_rrr_data,
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

import torch
from scipy.ndimage import gaussian_filter1d
import numpy as np
from tqdm import tqdm
import h5py

def main():
    # set config
    args = get_args()
    kwargs = {"model": "include:{}".format(args.model_config)}
    config = config_from_kwargs(kwargs)
    config = update_config(args.train_config, config)
    config = update_config(args, config)
    # set seed
    set_seed(config.seed)

    if args.input_mod =='me':
        input_mod = 'whisker-motion-energy'
    elif args.input_mod == 'of':
        input_mod = 'whisker-of'
    elif args.input_mod == 'of-2d':
        input_mod = 'whisker-of-video'
    elif args.input_mod == 'of-2d-v':
        input_mod = 'whisker-of-video'
    elif args.input_mod == 'all':
        # me, wheel_speed, choice, block
        input_mod = 'all'
    elif args.input_mod == 'other':
        # wheel_speed, choice, block
        input_mod = 'other'
    elif args.input_mod == 'of-all':
        input_mod = 'of-all'
    elif args.input_mod == 'ws':
        input_mod = 'wheel-speed'
    elif args.input_mod == 'whisker-video':
        input_mod = 'whisker-video'

    with open('data/eid.txt') as file:
        include_eids = [line.rstrip() for line in file]
        # include_eids = include_eids[:args.n_sessions]

    # select 100 idx from 120
    idx = np.random.choice(119, 100, replace=False)
    sorted_idx = np.sort(idx)

    train_data = {
        eid:
        {
            "X": [], # X[0] is train, X[1] is test, X[2] is val
            "y": [], # y[0] is train, y[1] is test, y[2] is val
            "timestamp": [], # timestamp[0] is train, timestamp[1] is test, timestamp[2] is val
            "setup": {}
        } 
        for eid in include_eids
    }
    # orig input, gt data, without spike smoothing
    for eid in include_eids:
        print("processing eid: ", eid)
        # set dataset
        dataset_split_dict = split_dataset(config.dirs.data_dir,eid=eid)
        train_dataloader, val_dataloader, test_dataloader = make_loader(config, dataset_split_dict)
        meta_data = get_metadata_from_loader(test_dataloader, config)
        print(f"meta_data: {meta_data}")
        train_X , train_y, train_timestamp = get_rrr_data(train_dataloader, input_mod)
        train_data[eid]["X"].append(train_X)
        train_data[eid]["y"].append(train_y)
        train_data[eid]["timestamp"].append(train_timestamp)
        test_X , test_y, test_timestamp = get_rrr_data(test_dataloader, input_mod)
        train_data[eid]["X"].append(test_X)
        train_data[eid]["y"].append(test_y)
        train_data[eid]["timestamp"].append(test_timestamp)
        val_X, val_y, val_timestamp = get_rrr_data(val_dataloader, input_mod)
        train_data[eid]["X"].append(val_X)
        train_data[eid]["y"].append(val_y)
        train_data[eid]["timestamp"].append(val_timestamp)
    # save data
    if args.input_mod == 'whisker-video':
        # Save using HDF5
        with h5py.File(f'/expanse/lustre/scratch/ywang74/temp_project/Downloads/data_rrr_{args.input_mod}.h5', 'w') as f:
            for eid, data in train_data.items():
                grp = f.create_group(str(eid))
                grp.create_dataset('X_train', data=data['X'][0], compression='gzip')
                grp.create_dataset('y_train', data=data['y'][0], compression='gzip')
                grp.create_dataset('timestamp_train', data=data['timestamp'][0], compression='gzip')
                grp.create_dataset('X_test', data=data['X'][1], compression='gzip')
                grp.create_dataset('y_test', data=data['y'][1], compression='gzip')
                grp.create_dataset('timestamp_test', data=data['timestamp'][1], compression='gzip')
                grp.create_dataset('X_val', data=data['X'][2], compression='gzip')
                grp.create_dataset('y_val', data=data['y'][2], compression='gzip')
                grp.create_dataset('timestamp_val', data=data['timestamp'][2], compression='gzip')
                # Store setup data in the group attributes
                for key, value in data['setup'].items():
                    print(key, value)
                    grp.attrs[key] = value
    else:
        np.save(f"data/data_rrr_{args.input_mod}.npy", train_data)

if __name__ == '__main__':
    main()
