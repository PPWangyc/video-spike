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
        input_mod = 'whisker-of-2d'
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

    with open('data/eid.txt') as file:
        include_eids = [line.rstrip() for line in file]
        # include_eids = include_eids[:args.n_sessions]
    
    # select 100 idx from 120
    idx = np.random.choice(119, 100, replace=False)
    sorted_idx = np.sort(idx)

    train_data = {
        eid:
        {
            "X": [], 
            "y": [], 
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
        train_X , train_y = get_rrr_data(train_dataloader, input_mod)
        train_data[eid]["X"].append(train_X)
        train_data[eid]["y"].append(train_y)
        test_X , test_y = get_rrr_data(test_dataloader, input_mod)
        train_data[eid]["X"].append(test_X)
        train_data[eid]["y"].append(test_y)

    # save data
    np.save(f"data/data_rrr_{args.input_mod}.npy", train_data)
    
    
    

if __name__ == '__main__':
    main()
